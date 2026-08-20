#pragma once

// Lifted out of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions that
// operate on them can live in their own translation units. These are INTERNAL to the
// recompiler: nothing outside src/gpu/recompiler/ should include this header.

// rdna2_to_spirv.cpp — see rdna2_to_spirv.hpp. Internal SpirvCompute builder + the VALU translator.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "rdna2_to_spirv_internal.hpp"
#include "rdna2_alu_support.hpp"

namespace prosper::gpu {


inline bool sopp_is_noop(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOPP) return false;
    switch (in.opcode) {
        case 0x00:   // s_nop
        case 0x0c:   // s_waitcnt
        case 0x20:   // s_inst_prefetch
        case 0x21:   // s_clause
        case 0x22:   // s_wait_idle
            // NOTE: s_waitcnt_vscnt is NOT SOPP on gfx10 — it is SOPK opcode 0x17
            // (round-trip: llvm-mc gfx1010 encodes it 0xBBFD0000). Handled in the SOPK case.
            return true;
        default:
            return false;
    }
}
uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in);
uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source);

// What a surviving read of R is allowed to be.
//
// AnyRead     -- R must not be read at all before it is redefined. The original and default proof.
// MaskDomainOnly -- R may be read as 32-bit scalar DATA, but not by a lane-mask consumer. Only
//                meaningful for a VCC half, and only sound because prosper models a VCC half written
//                by a B32 scalar op as scalar data with its own storage: a 32-bit read of that half
//                reproduces the written dword exactly, while a MASK read would need the per-lane
//                predicate the scalar model deliberately does not reconstruct.
//
// The distinction is 32-bit-ness, not explicitness, and that is the whole subtlety: a VOP3
// `v_cndmask_b32 v0, v1, v2, vcc` names VCC explicitly and still reads the mask domain. Every
// mask consumer reads the PAIR, so requiring `scalar_alu_source_words(...) == 1` excludes all of
// them -- VOP3 cndmask, VOP3B carry-in, `s_mov_b64 exec, vcc`, and B64 mask logic alike -- without
// enumerating them. The e32 implicit readers and the vccz/vccnz branches are rejected earlier and
// unconditionally, so they are excluded on both paths.
enum class ScalarMergeProof { AnyRead, MaskDomainOnly };

// Why the proof failed, for the reject message. A liveness proof that reports only "failed" makes
// every widening of it a guess: the first attempt at MaskDomainOnly did not accept the GTA V kernel
// it was written for, and without the blocking pc there was no way to tell whether the walk stopped
// at a mask read, a data read, or an unmodelled opcode -- three different follow-ups.
struct ScalarMergeBlocker {
    uint32_t pc = UINT32_MAX;
    const char* kind = "none";
};

inline bool sgpr_dead_at_merge(const std::vector<Rdna2Inst>& ins, uint32_t target, int R,
                               ScalarMergeProof proof = ScalarMergeProof::AnyRead,
                               ScalarMergeBlocker* blocker = nullptr) {
    const auto block = [&](const Rdna2Inst& at, const char* kind) {
        if (blocker && blocker->pc == UINT32_MAX) { blocker->pc = at.pc; blocker->kind = kind; }
        return false;
    };
    // MaskDomainOnly is a VCC-half question by construction: an ordinary SGPR has no mask domain,
    // so the relaxation would silently become "reads are fine", which is not a proof of anything.
    if (proof == ScalarMergeProof::MaskDomainOnly && R != 106 && R != 107)
        return false;
    const bool data_read_ok = proof == ScalarMergeProof::MaskDomainOnly;
    // Prove liveness over the actual scalar CFG, not just lexical order. UE4 commonly places another
    // forward EXECZ after an if/merge and only then overwrites the scratch SGPR/VCC value. A linear
    // scan used to reject that safe shape merely because it encountered the second branch. Explore
    // BOTH successors of every recognized conditional branch; a read on either path fails the proof,
    // while a redefinition/end terminates only that path. The visited set also bounds back-edges.
    std::unordered_map<uint32_t, size_t> pc_to_index;
    for (size_t i = 0; i < ins.size(); ++i) pc_to_index.emplace(ins[i].pc, i);
    const auto start = pc_to_index.find(target);
    if (start == pc_to_index.end()) return false;
    std::vector<size_t> pending{start->second};
    std::vector<uint8_t> visited(ins.size(), 0);
    while (!pending.empty()) {
        size_t index = pending.back();
        pending.pop_back();
        if (index >= ins.size() || visited[index]) continue;
        visited[index] = 1;
        const auto& in = ins[index];
        if (in.is_end) continue;                           // this path ends without a read
        // VCC (R = 106/107) has IMPLICIT readers the operand scan can't see: the e32 VOP2 carry /
        // cndmask ops (0x01, 0x28-0x2A) and the vccz/vccnz branches. Flag those as reads up front so
        // the redef checks below are sound for VCC too (which the s_mov carve-out excludes, but the
        // SMEM carve-out needs — compilers love `s_buffer_load_dword vcc_lo, …` scratch loads).
        if (R == 106 || R == 107) {
            if (in.fmt == Rdna2Format::VOP2 &&
                (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2A)))
                return block(in, "vop2-implicit-vcc");
            if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07))
                return block(in, "vccz-branch");
        }
        switch (in.fmt) {
            // Most SOPK instructions remain fail-closed: s_addk/s_mulk/s_cmovk/s_cmpk read or
            // read-modify-write their encoded "dst" while decode exposes no source. MOVK_I32 is a
            // pure one-word definition and WAITCNT_VSCNT is register-transparent; GTA's low-only
            // VCC lifetime crosses both before the actual pair replacement.
            case Rdna2Format::SOPK:
                if (in.opcode == kSopkOpcodeWaitcntVscnt &&
                    in.dst.kind == OperandKind::SGPR && in.dst.value == 125)
                    break;
                if (in.opcode == kSopkOpcodeMovkI32 && in.dst.kind == OperandKind::SGPR) {
                    if (in.dst.value == R) continue;
                    break;
                }
                // Under MaskDomainOnly the whole read-modify-write SOPK family is admissible. Every
                // SOPK operates on ONE dword: s_addk_i32/s_mulk_i32 read and rewrite their encoded
                // destination, s_cmpk_* read it and write SCC, s_cmovk_i32 reads it and may rewrite
                // it. All three are 32-bit scalar-DATA touches of the half; none can consume it as a
                // 64-bit lane mask, which is the only thing this proof needs to exclude.
                //
                // Deliberately does NOT claim the half died, even for the forms that always write
                // it: continuing the walk is the conservative direction, and a later genuine mask
                // read still fails the proof. Being wrong about the kill would be unsound; being
                // silent about it only costs acceptances.
                if (data_read_ok && in.dst.kind == OperandKind::SGPR) break;
                return block(in, "unmodelled-sopk");
            case Rdna2Format::SOPP:
                // Hint/sync SOPPs read and write nothing — scan through them (s_waitcnt is ubiquitous
                // after the merge). A barrier is also scalar-register-transparent; it changes only
                // workgroup execution order. For branches, enqueue every possible successor. Unknown
                // scalar control flow remains fail-closed.
                if (sopp_is_noop(in) || in.opcode == 0x0a) break;
                if (in.opcode == 0x02 ||
                    (in.opcode >= 0x04 && in.opcode <= 0x09)) {
                    const int64_t branch_pc = static_cast<int64_t>(in.pc) +
                        in.len_dwords + in.simm16;
                    const auto branch = branch_pc >= 0
                        ? pc_to_index.find(static_cast<uint32_t>(branch_pc)) : pc_to_index.end();
                    if (branch == pc_to_index.end()) return false;
                    pending.push_back(branch->second);
                    if (in.opcode != 0x02) {
                        if (index + 1 >= ins.size()) return false;
                        pending.push_back(index + 1);
                    }
                    continue;
                }
                return block(in, "unmodelled-sopp");
            case Rdna2Format::SMEM: {
                // s_load/s_buffer_load: reads the SBASE pair (src[0], 2 regs) + SOFFSET (src[1]);
                // redefines N consecutive dst SGPRs. Anything not a plain load -> bail.
                uint32_t n = 0;
                switch (in.opcode) {
                    case 0x0: case 0x8: n = 1;  break;   case 0x1: case 0x9: n = 2;  break;
                    case 0x2: case 0xA: n = 4;  break;   case 0x3: case 0xB: n = 8;  break;
                    case 0x4: case 0xC: n = 16; break;
                    default: return block(in, "unmodelled-smem");
                }
                if (in.src[0].value == R || in.src[0].value + 1 == R)
                    return block(in, "smem-sbase");                                          // SBASE read
                // SOFFSET is one dword and is consumed as an address, never as a lane mask.
                if ((in.src[1].kind == OperandKind::SGPR || in.src[1].kind == OperandKind::Special) &&
                    in.src[1].value == R && !data_read_ok)
                    return block(in, "smem-soffset");                                        // SOFFSET read
                if (R >= in.dst.value && R < in.dst.value + (int)n) continue;  // redefined: this path is dead
                break;
            }
            case Rdna2Format::SOP1: case Rdna2Format::SOP2: case Rdna2Format::SOPC:
            case Rdna2Format::VOP1: case Rdna2Format::VOP2: case Rdna2Format::VOP3: case Rdna2Format::VOPC:
            case Rdna2Format::DS:    // DS operands are VGPR/M0 only; it cannot read an ordinary SGPR/VCC
            case Rdna2Format::EXP: { // EXP data sources are all VGPRs — it can never read an SGPR
                // Conditional moves, SOPK immediate operations, and SOP1 bitsets may read their
                // encoded destination before replacing it. Their decoded source list does not
                // contain that old value, so consult the shared implicit-read inventory first.
                const uint32_t implicit_read_width =
                    scalar_implicit_destination_read_width(in);
                if (implicit_read_width && R >= in.dst.value &&
                    R < in.dst.value + static_cast<int>(implicit_read_width) &&
                    !(data_read_ok && implicit_read_width == 1))
                    return block(in, "implicit-dst-rmw");
                for (int k = 0; k < in.n_src; k++) {
                    if (in.src[k].kind != OperandKind::SGPR &&
                        in.src[k].kind != OperandKind::Special) continue;
                    // Operand does not carry B32/B64 width. Use the shared opcode inventory so a
                    // physical low-half read (notably GTA's CSELECT_B32 -> V_ADD3 chain) cannot
                    // falsely keep the sibling live, while unlisted VOP3 sources remain pairs.
                    const uint32_t words = scalar_alu_source_words(in, k);
                    if (words == UINT32_MAX) continue;
                    if (R >= in.src[k].value &&
                        R < in.src[k].value + static_cast<int>(words)) {
                        // A one-dword source is a scalar-data read. Anything wider reads the pair,
                        // which is exactly the set of lane-mask consumers.
                        if (data_read_ok && words == 1) continue;
                        return block(in, words == 1 ? "source-dword" : "source-pair");
                    }
                }
                // A VOP3B carry-out (sdst) is a 64-bit mask write — reads none of R beyond its sources.
                if (in.sdst.kind == OperandKind::SGPR &&
                    (in.sdst.value == R || in.sdst.value + 1 == R)) continue;   // redefined (pair)
                // A non-X VOPC e32 compare writes the whole VCC pair — kills both halves. V_CMPX
                // writes EXEC ONLY on RDNA2 (ISA 12.9: "EXEC[threadId] = ..."; no VCC/SDST dest),
                // so it must NOT count as a VCC kill — miscounting it let safe_execz linearize a
                // block whose vcc scratch-write hardware would have skipped.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                    (in.dst.value == 106 || in.dst.value == 107) && (R == 106 || R == 107))
                    continue;
                // A scalar redefinition kills every word covered by its opcode's write width. Keep
                // this after the explicit/implicit read checks above: B64 read-modify-write forms
                // must observe the old pair before replacing it. EXEC/M0 (124/126/127) still cannot
                // be proven dead (implicit reads everywhere), and cmpx writes EXEC rather than its
                // decoded SGPR destination.
                const uint32_t dst_width = scalar_write_width(in);
                if (dst_width && in.dst.kind == OperandKind::SGPR && R <= 107 &&
                    R >= in.dst.value && R < in.dst.value + static_cast<int>(dst_width) &&
                    !(in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)))
                    continue;
                break;
            }
            case Rdna2Format::MIMG:
                // Image packets fully decode both scalar descriptor operands: the eight-dword T#/U#
                // at src[1] and the four-dword S# at src[2]. Some opcodes do not consume the sampler,
                // but accounting for both ranges is deliberately conservative and lets the proof scan
                // through every MIMG without needing a second opcode inventory.
                if (in.src[1].kind != OperandKind::SGPR ||
                    in.src[2].kind != OperandKind::SGPR) return false;
                if (R >= in.src[1].value && R < in.src[1].value + 8) return false;
                if (R >= in.src[2].value && R < in.src[2].value + 4) return false;
                break;
            case Rdna2Format::MUBUF:
            case Rdna2Format::MTBUF:
                // Buffer packets fully decode their scalar inputs: SRSRC is the four-dword
                // descriptor at src[1], and SOFFSET is the single src[2] operand.  Treating every
                // buffer access as an unknowable SGPR read made an otherwise-dead VCC_HI scratch
                // word look live across Astro Bot's world-map setup pass.  Keep the proof exact by
                // checking the complete descriptor range and explicit offset before scanning on.
                if (in.src[1].kind != OperandKind::SGPR) return false;
                if (R >= in.src[1].value && R < in.src[1].value + 4) return false;
                if ((in.src[2].kind == OperandKind::SGPR ||
                     in.src[2].kind == OperandKind::Special) && in.src[2].value == R)
                    return false;
                break;
            case Rdna2Format::FLAT:
                // FLAT/GLOBAL/SCRATCH packets keep every data/address operand in VGPRs except
                // SADDR, the optional scalar address pair at src[1]. NULL/off (125) reads no
                // scalar register. An LDS transfer also observes M0 implicitly, so keep that
                // uncommon form fail-closed. Account for the complete SADDR pair conservatively;
                // this liveness proof does not imply that the memory operation is translatable.
                if (in.flat_lds) return false;
                if ((in.src[1].kind == OperandKind::SGPR ||
                     in.src[1].kind == OperandKind::Special) &&
                    in.src[1].value != 125 &&
                    (in.src[1].value == R || in.src[1].value + 1 == R))
                    return false;
                break;
            default:
                return false;   // memory/branch/interp/unknown: can't bound reads of R -> assume live
        }
        if (index + 1 < ins.size()) pending.push_back(index + 1);
    }
    return true;   // every reachable path hit a redefinition/end without a read
}

inline std::unordered_set<uint32_t> proven_cselect_b64_low_only_pcs(
        const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> result;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) break;
        // GTA V 0x413cf9a00 pc60 selects two scalar-data pairs into VCC, then reads only
        // VCC_LO as one scalar dword. Its second alternative currently has only the low word
        // available (the zero-record S_BUFFER load at pc16 writes s16, not s17). Admit that
        // incomplete-source form only for the exact ordinary-SGPR-to-VCC packet shape and only
        // when the shared CFG liveness walk proves VCC_HI dead from the following instruction.
        // An implicit VCC consumer counts as a high-half read in that walk, so no per-lane mask
        // can observe the unavailable selected word.
        if (in.fmt == Rdna2Format::SOP2 && in.opcode == 0x0bu &&
            in.dst.kind == OperandKind::SGPR && in.dst.value == 106 &&
            in.src[0].kind == OperandKind::SGPR &&
            in.src[1].kind == OperandKind::SGPR &&
            sgpr_dead_at_merge(ins, in.pc + in.len_dwords, 107))
            result.insert(in.pc);
    }
    return result;
}

// A one-dword scalar write to Wave64 VCC_LO normally has to preserve VCC_HI and rebuild the
// architectural per-lane predicate from both physical words. GTA V also recycles only the low word
// as scalar scratch, then replaces the complete pair before anything can observe the untouched high
// word. The scalar-data-vs-mask decision remains path-local in emit_alu; this predicate only names
// B32 packets for which a dead-high proof can make that scalar path discard the old predicate.
inline bool is_wave64_vcc_lo_scalar_b32_candidate(const Rdna2Inst& in) {
    return is_gtav_wave64_vcc_lo_scalar_cselect(in) ||
        (in.fmt == Rdna2Format::SOP2 && sop2_is_b32_logical(in.opcode) &&
         in.dst.kind == OperandKind::SGPR && in.dst.value == 106);
}

inline std::unordered_set<uint32_t> proven_wave64_vcc_b32_low_only_pcs(
        const std::vector<Rdna2Inst>& ins, bool include_logical = true) {
    std::unordered_set<uint32_t> result;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) break;
        const bool candidate = include_logical
            ? is_wave64_vcc_lo_scalar_b32_candidate(in)
            : is_gtav_wave64_vcc_lo_scalar_cselect(in);
        if (candidate &&
            sgpr_dead_at_merge(ins, in.pc + in.len_dwords, 107))
            result.insert(in.pc);
    }
    return result;
}
bool vcc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);

// WATERFALL loops (#273 — DOLL's skinned scene VS): the readfirstlane-uniformize idiom
//   L: v_readfirstlane s4, vIDX ; v_cmpx_eq_u32 s4, vIDX ; s_mov m0, s4 ; v_movrels …
//      s_andn2_b64 REMAIN, REMAIN, exec ; s_mov_b64 exec, REMAIN ; s_cbranch_scc1 L
// iterates once per DISTINCT per-lane index, with EXEC selecting the matching lanes. In the
// per-invocation model each lane's iteration IS its own: readfirstlane returns MY value, the cmpx
// keeps me active, and the andn2 leaves MY remaining-mask empty — so the loop body runs EXACTLY
// ONCE per invocation and the backward branch (whose SCC = "any lane remains", a cross-lane
// reduction) is never taken. A BACKWARD s_cbranch_scc0/scc1 whose SCC was produced by a 64-bit
// wave-mask op is therefore safe to DROP (linearize), exactly like the forward kill-mask branches.
// The SCC producer may sit a few instructions back (the shape interposes `s_mov_b64 exec, REMAIN`,
// which does not write SCC) — walk back over non-SCC-writing instructions to find it.
// CONFIDENCE: MED — exec-diff kernel T19 locks the once-through semantics.
inline std::unordered_set<uint32_t> waterfall_branches(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> out;
    for (size_t i = 0; i < ins.size(); i++) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || (in.opcode != 0x04 && in.opcode != 0x05) || in.simm16 >= 0)
            continue;
        // Walk back to the most recent SCC-writing instruction, skipping ops known not to touch SCC
        // (scalar moves, all vector ALU incl. VOPC — it writes VCC/EXEC, not SCC — memory, hints).
        for (size_t j = i; j-- > 0;) {
            const Rdna2Inst& p = ins[j];
            bool skip_ok =
                (p.fmt == Rdna2Format::SOP1 && (p.opcode == 0x03 || p.opcode == 0x04)) ||   // s_mov_b32/b64
                p.fmt == Rdna2Format::VOP1 || p.fmt == Rdna2Format::VOP2 || p.fmt == Rdna2Format::VOP3 ||
                p.fmt == Rdna2Format::VOPC || p.fmt == Rdna2Format::MIMG ||
                p.fmt == Rdna2Format::MUBUF || p.fmt == Rdna2Format::MTBUF ||
                p.fmt == Rdna2Format::FLAT ||
                sopp_is_noop(p);
            if (skip_ok) continue;
            // The SCC producer: a 64-bit wave-mask logical op (s_and/or/xor/andn2_b64, SCC = result!=0).
            if (p.fmt == Rdna2Format::SOP2 &&
                (p.opcode == 0x0f || p.opcode == 0x11 || p.opcode == 0x13 || p.opcode == 0x15) &&
                p.dst.kind == OperandKind::SGPR && p.dst.value <= 106)
                out.insert(in.pc);
            break;   // any other SCC producer (s_cmp etc.): a REAL loop condition — not a waterfall
        }
    }
    return out;
}

// A recognized COUNTED loop (the game's MSAA-resolve / accumulation shape). Two accepted shapes:
//   (a) TOP-tested: a single backward UNCONDITIONAL s_branch (the back-edge) to a header, with
//       exactly one forward s_cbranch_scc0/scc1 exit inside the body — `for (...) {...}`.
//   (b) BOTTOM-tested (do-while): the back-edge is ITSELF a backward s_cbranch_scc0/scc1; the loop
//       continues while its SCC condition holds and exits by falling through — `do {...} while(...)`.
//       This is what the compiler emits for a fixed-trip texture-accumulation loop whose whole body
//       (including the s_cmp that sets the exit SCC) runs before the test. Alex Kidd's (PPSA02664)
//       light/blur-accumulation pixel shaders use it, and rejecting it dropped their draws (#320).
// Both are lowered by ONE emitter: shape (b) is modeled as exit_branch_pc == backedge_pc, so the
// emitter emits the entire body as the "condition region" (runs every iteration incl. the exiting
// one — exactly do-while), tests SCC at the bottom, and its "body region" (exit_branch+1..backedge)
// is empty. Anything more complex (nested loops, VCC/EXECNZ exits, multiple back-edges, mid-loop
// s_branch) is rejected — the recompiler then falls back to the straight-line path (as before).
struct CountedLoop {
    bool found = false;
    uint32_t header_pc = 0;       // loop header (target of the back-edge; condition eval starts here)
    uint32_t exit_branch_pc = 0;  // the s_cbranch_scc0/scc1 exit test (== backedge_pc for a do-while)
    uint32_t backedge_pc = 0;     // the backward branch (unconditional s_branch, or the scc back-edge)
    uint32_t exit_pc = 0;         // first pc after the loop (== backedge_pc + its length)
    bool exit_on_scc0 = true;     // exit-test polarity: true ⇔ the loop continues while SCC==1 (exits on SCC==0)
};
inline uint32_t branch_target(const Rdna2Inst& in) { return in.pc + in.len_dwords + (uint32_t)(int32_t)in.simm16; }

inline CountedLoop detect_counted_loop(const std::vector<Rdna2Inst>& ins) {
    CountedLoop L;
    // The single backward branch that closes the loop. A conditional scc0/scc1 back-edge is a
    // bottom-tested do-while; an unconditional s_branch is a top-tested loop.
    // A backward s_cbranch_scc0/scc1 whose SCC is a 64-bit wave-mask reduction is a WATERFALL
    // (once-through per invocation), NOT a loop back-edge — its cross-lane SCC has no representable
    // per-lane value, and every emit path already consumes it via safe_branches/the linearizer.
    // Waterfalls are excluded from the back-edge count entirely: SELECTING one would feed the loop
    // emitter a poisoned SCC exit and mis-lower the divergent body (T19), and merely COUNTING one
    // would reject a counted loop that legitimately coexists with a waterfall elsewhere in the same
    // shader — a previously-recompilable combination (kernel 44d locks it).
    const std::unordered_set<uint32_t> wf = waterfall_branches(ins);
    const Rdna2Inst* back = nullptr; int nback = 0;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || in.simm16 >= 0) continue;
        if (in.opcode == 0x02 ||
            ((in.opcode == 0x04 || in.opcode == 0x05) && !wf.count(in.pc))) { back = &in; nback++; }
    }
    if (nback != 1) return L;                       // 0 -> no loop; >1 -> nested/multiple (unhandled)
    const bool do_while = (back->opcode != 0x02);   // conditional back-edge => bottom-tested
    const uint32_t header = branch_target(*back);
    bool header_ok = false;
    for (const auto& in : ins) if (in.pc == header && in.pc < back->pc) { header_ok = true; break; }
    if (!header_ok) return L;
    const uint32_t exit_pc = back->pc + back->len_dwords;
    const Rdna2Inst* exitbr = nullptr; int nexit = 0;
    for (const auto& in : ins) {                    // scan the loop body [header, back-edge]
        if (in.pc < header || in.pc > back->pc || in.fmt != Rdna2Format::SOPP) continue;
        if (&in == back) continue;                  // the back-edge is classified above, not here
        switch (in.opcode) {
            case 0x04: case 0x05:                   // s_cbranch_scc0 / scc1 — must be the single exit
                if (branch_target(in) != exit_pc) return L;
                exitbr = &in; nexit++; break;
            case 0x02:                              // any other s_branch -> reject
                return L;
            case 0x06: case 0x07: case 0x09:        // vcc / execnz branches -> reject
                return L;
            case 0x08: default: break;              // execz (forward, predication-handled) / hints ok
        }
    }
    if (do_while) {
        // Bottom-tested: the conditional back-edge IS the exit test; no separate forward exit is
        // allowed (a do-while with an inner uniform-if stays unsupported and rejects, conservatively).
        // The back-edge is TAKEN (loops) while its SCC condition holds. exit_on_scc0 drives the
        // emitter's loop_cond = (exit_on_scc0 ? SCC : !SCC), i.e. "continue when this is true": an
        // s_cbranch_scc1 back-edge continues while SCC==1 (exit_on_scc0=true); scc0 continues while
        // SCC==0 (exit_on_scc0=false).
        if (nexit != 0) return L;
        L.found = true; L.header_pc = header; L.exit_branch_pc = back->pc; L.backedge_pc = back->pc;
        L.exit_pc = exit_pc; L.exit_on_scc0 = (back->opcode == 0x05);
        return L;
    }
    if (nexit != 1) return L;
    L.found = true; L.header_pc = header; L.exit_branch_pc = exitbr->pc; L.backedge_pc = back->pc;
    L.exit_pc = exit_pc; L.exit_on_scc0 = (exitbr->opcode == 0x04);
    return L;
}

// MULTI-LOOP EXEC/VCC/SCC EXITS (#273/#615/#1554 — post-process and light-accumulation PSes,
// title-composite content producers, and compute histogram loops). The divergent compiled shape:
//     header: <exec recompute: v_cmpx_.. counter,bound  |  v_cmp..;s_andn2_b64 exec,exec,vcc>
//             s_cbranch_execz EXIT                      ; leave when no lane remains
//     body:   ... (nested forward-execz if regions, saveexec/restore, scalar counter++) ...
//             s_branch header                           ; backward unconditional back-edge
//     EXIT:   s_mov_b64 exec, <saved>
// Fragment semantics are exact: a forced 64-lane native subgroup vote keeps the complete guest wave
// in the loop until every EXEC/VCC bit clears. Cleared lanes remain synchronization participants while
// EXEC predicates their vector writes, so scalar state advances exactly as on RDNA2. Vertex and the
// narrow structured-compute path retain the older per-invocation lowering and its uniformity guards.
//
// A second flavor (DOLL's scalar-indexed unroll): the back-edge is a backward s_cbranch_EXECNZ and
// the header IS the execz exit (empty condition region); the body may carry an extra forward
// vccz/execz BREAK to the exit. A break lowers as a plain forward IF over the remainder of the body
// (skip-to-backedge): the broken lane's EXEC bit is already clear (the compiled break condition
// mirrors the exec recompute), so the next header check exits it. Fragment wave64 additionally
// supports a VCCZ break with an unconditional back-edge by carrying the uniform continue vote to the
// loop latch and branching the complete wave directly to the loop merge.
struct DivLoop {
    enum class Condition : uint8_t { Exec, Vcc, Scc };
    uint32_t header_pc = 0;        // back-edge target; condition region = [header_pc, exit_branch_pc)
    uint32_t exit_branch_pc = 0;   // canonical forward execz/vccz/scc branch whose target is exit_pc
    uint32_t backedge_pc = 0;      // backward s_branch (unconditional) or s_cbranch_execnz
    uint32_t exit_pc = 0;          // backedge_pc + its length (first pc after the loop)
    std::vector<uint32_t> break_pcs;   // extra forward vccz/execz -> exit_pc (lowered as body ifs)
    bool direct_exec_breaks = false;   // unconditional back-edge: an interior execz exits directly
    bool direct_wave_breaks = false;   // fragment wave64: an interior vccz exits the complete wave
    bool bottom_tested = false;        // s_cbranch_execnz back-edge is the condition (do-while)
    Condition condition = Condition::Exec;
    // EXECZ/VCCZ and SCC0 all continue while their represented predicate is set. SCC1 is the one
    // admitted opposite-polarity canonical exit: it leaves the loop while SCC is set.
    bool continue_on_set = true;
};

// A scalar VCCZ branch tests whether ANY active lane set VCC, not this lane's bit. Stages without
// fragment's forced wave64 vote may lower it as structured control only when the VCC producer is
// provably uniform across the wave. Dead Cells' light loops use the narrow compiler shape
//   v_cvt_i32_f32 vBOUND, sBOUND; ...; v_cmp_* vcc, sCOUNTER, vBOUND; s_cbranch_vccz EXIT
// so every active lane makes the same comparison. Unity also derives the bound through a short
// lane-local VALU chain fed only by scalar sources. Trace that chain backwards: uniform inputs stay
// uniform through ordinary VOP1/VOP2 arithmetic, while lane permutations, VCC-dependent selects,
// carry chains, interpolants, and unresolved VGPR inputs remain rejected.
// True when `in` provably cannot overwrite VCC — used by vcc_exit_is_wave_uniform's compare-finding
// walk (#590) to look past instructions scheduled between the VOPC and its consuming branch (real
// UE4 compute kernels hoist ~15 unrelated ALU ops in between). Anything not provably safe stops the
// walk conservatively: VOPC itself (the implicit dst), the VOP2 carry-chain ops (v_*_co_ci_u32,
// 0x28-0x2a), VOP3 windows that may carry a scalar dest (VOPC-as-VOP3 sdst at 0x000-0x13f, VOP3B
// carry-out / v_readlane at 0x300+ — the decoder does not expose their sdst), and any scalar or
// SMEM write whose dest range could cover SGPR 106/107 (the shaders use s106/s107 as scratch).
static bool cannot_write_vcc(const Rdna2Inst& in) {
    auto sgpr_dst_misses_vcc = [&](uint32_t n_dwords) {
        if (in.dst.kind != OperandKind::SGPR && in.dst.kind != OperandKind::Special) return true;
        const int lo = in.dst.value, hi = in.dst.value + (int)n_dwords - 1;
        return hi < 106 || lo > 107;
    };
    switch (in.fmt) {
        case Rdna2Format::VOP1:
        case Rdna2Format::VINTRP:
        case Rdna2Format::MUBUF: case Rdna2Format::MTBUF: case Rdna2Format::MIMG:
        case Rdna2Format::DS:    case Rdna2Format::FLAT:
        case Rdna2Format::EXP:                                     // VGPR/memory dsts only
        case Rdna2Format::SOPP:                                    // branches/hints/waitcnt: no dst
        case Rdna2Format::SOPC:  return true;                      // writes SCC only
        case Rdna2Format::VOP2:  return in.opcode < 0x28 || in.opcode > 0x2a;
        case Rdna2Format::VOP3P: return true;                      // packed VALU: VGPR dst, no carry
        case Rdna2Format::VOP3:
            // gfx10 v_writelane_b32 writes only its VGPR lane slot. v_readlane_b32 writes the
            // scalar destination encoded in the decoder's dst field, so it is VCC-preserving as
            // long as that destination does not overlap s106:s107. UE4 schedules long scalar-spill
            // sequences of these two ops between a uniform VOPC and its s_cbranch_vccz.
            if (in.opcode == 0x361) return true;
            if (in.opcode == 0x360) return sgpr_dst_misses_vcc(1);
            if (in.opcode == 0x176)
                return in.sdst.value != 106 && in.sdst.value != 107; // v_mad_u64_u32 carry-out
            // The remaining VOP3B ops inside the plain-VALU window — v_div_scale_f32/f64 (0x16D/
            // 0x16E) and v_mad_i64_i32 (0x177) — write a scalar carry/flag SDST that may be VCC;
            // their VGPR dst would otherwise satisfy sgpr_dst_misses_vcc. Stop the walk for them.
            if (in.opcode == 0x16D || in.opcode == 0x16E || in.opcode == 0x177) return false;
            return in.opcode >= 0x140 && in.opcode < 0x300 &&
                   sgpr_dst_misses_vcc(1);                         // plain-VALU window only
        case Rdna2Format::SOP1: case Rdna2Format::SOP2: case Rdna2Format::SOPK:
            return sgpr_dst_misses_vcc(2);                         // conservative 64-bit pair
        case Rdna2Format::SMEM: {
            uint32_t n = 0;
            switch (in.opcode & 7) { case 0: n = 1; break; case 1: n = 2; break; case 2: n = 4; break;
                                     case 3: n = 8; break; case 4: n = 16; break; default: return false; }
            return sgpr_dst_misses_vcc(n);
        }
        default: return false;                                     // VOPC / unknown: stops the walk
    }
}

inline bool vcc_exit_is_wave_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc) {
    const Rdna2Inst* compare = nullptr;
    for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
        if (it->pc >= branch_pc) continue;
        if (sopp_is_noop(*it)) continue;
        if (it->fmt == Rdna2Format::VOPC) { compare = &*it; break; }
        if (cannot_write_vcc(*it)) continue;   // (#590) look past provably-VCC-preserving instrs —
        break;                                 // VCC's value is frozen at the compare; only a real
                                               // rewrite between compare and branch invalidates it
    }
    if (!compare || compare->fmt != Rdna2Format::VOPC || vopc_is_cmpx(compare->opcode)) return false;

    std::function<bool(const Operand&, uint32_t)> uniform_operand_at;
    uniform_operand_at = [&](const Operand& operand, uint32_t use_pc) -> bool {
        if (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special ||
            operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat ||
            operand.kind == OperandKind::Literal)
            return true; // scalar-register sources (including VCC halves) broadcast one wave value
        if (operand.kind != OperandKind::VGPR) return false;
        for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
            if (it->pc >= use_pc) continue;
            if (operand.value == rdna2_tfe_status_vgpr(*it)) return false;
            const uint32_t writes = rdna2_vgpr_write_count(*it);
            if (!writes || operand.value < it->dst.value ||
                operand.value >= it->dst.value + (int32_t)writes) continue;
            if (it->dst.value != operand.value || it->has_dpp) return false;
            // The write makes every currently active lane uniform. Keep the original conservative
            // EXEC rule: any later mask change could expose a lane that retained an older value.
            for (const auto& between : ins)
                if (between.pc > it->pc && between.pc < compare->pc &&
                    rdna2_instruction_may_change_exec(between)) return false;

            bool pure_lane_alu = false;
            if (it->fmt == Rdna2Format::VOP1) {
                pure_lane_alu = it->n_src == 1;
            } else if (it->fmt == Rdna2Format::VOP2) {
                // v_cndmask consumes VCC; the carry family consumes/produces VCC. Neither preserves
                // scalar-source uniformity as a simple lane-local expression.
                pure_lane_alu = it->n_src == 2 && it->opcode != 0x01 &&
                    (it->opcode < 0x28 || it->opcode > 0x2a);
            }
            if (!pure_lane_alu) return false;
            for (uint32_t i = 0; i < it->n_src; ++i)
                if (!uniform_operand_at(it->src[i], it->pc)) return false;
            return true;
        }
        return false;
    };
    for (uint32_t i = 0; i < compare->n_src; ++i)
        if (!uniform_operand_at(compare->src[i], compare->pc)) return false;
    return compare->n_src != 0;
}

// Returns the loops in header-pc order, or {} when any backward branch doesn't fit the shape.
// `safe` carries the linearized branches (waterfalls etc.) which are not loop back-edges.
//
// An empty result has TWO causes that the caller cannot tell apart, and neither can the log: no
// eligible backward branch was found at all (the `out.empty()` early return below, which reaches
// none of the reject sites), or a shape check refused. Callers conflate them deliberately --
// `(cf_rejected || Ls.empty())` -- so this distinction matters only to a reader of the diagnostic,
// which is why the reject reporter below prints on exactly one of those two paths.
inline std::vector<DivLoop> detect_divergent_loops(const std::vector<Rdna2Inst>& ins,
                                            const std::unordered_set<uint32_t>& safe,
                                            bool exact_fragment_wave_breaks = false,
                                            const RecompileDiagnosticContext& diagnostic = {},
                                            const char* caller_role = "emit") {
    std::vector<DivLoop> out;
    // PROSPER_DBG: name WHICH rejection discards the loops. Every `return {}` here throws away
    // EVERY loop in the shader, not just the one that failed. What the caller then does with an
    // empty list is NOT fixed -- see the note on `caller_role` below -- but one of the outcomes is
    // the whole-stream CFG dispatcher, which EMULATES the loop instead of emitting a real SPIR-V
    // one. So one unhandled shape silently changes how an entire program executes. Static reading can
    // narrow a symptom to this function in minutes and then stall, because the sites are
    // indistinguishable from outside; this makes them distinguishable.
    //
    // Do NOT read an absent line as "this function was never called for that program". Absence has
    // three other causes: the `out.empty()` early return (no eligible backward branch, so no site is
    // reached), a caller that supplies no diagnostic context (suppressed below), and PROSPER_DBG
    // itself, which desyncs a timing-dependent pad script badly enough that a routed run may never
    // reach the dispatch you wanted to observe (see the note at the top of this file). Whether a
    // given caller consults this function at all is a question for the CALL SITE, not for the log.
    // Reported ONCE per (program, cause, pc). A program is recompiled or re-looked-up many times
    // over a run, so an unfiltered print repeats the same rejection -- measured on a GTA V gameplay
    // run, the busiest single site emitted 70 lines. A log whose line count far exceeds its finding
    // count invites exactly one mistake: quoting the line count as a finding count. De-duplicating
    // here means the raw count is the honest statistic.
    //
    // Reported ONLY when the caller supplied a real program address. Both graphics stages have no
    // diagnostic parameter at all and so stay silent: they would report `program=0x0`, which is
    // wrong twice over -- the address is not a program, and the de-dup key would merge genuinely
    // different shaders rejecting at the same cause and pc into one line, making a count of
    // distinct programs read LOW. An unattributed line is not a weaker finding, it is a wrong one.
    //
    // The message states ONLY what is certain at this point, and deliberately says nothing about
    // what happens next. It is tempting to add "and the stream then falls back to the CFG
    // dispatcher" -- an earlier version did, and so did the first attempt at THIS comment, which
    // asserted the opposite outcome just as confidently and was equally wrong.
    //
    // The honest statement is that the outcome is not determined here, and this comment will not
    // enumerate the ways it can go -- deliberately, because two drafts already got an enumeration
    // wrong in opposite directions.
    //
    // What is checkable and sufficient: an empty list is ONE input among several to a decision taken
    // in `emit_body`, whose branches carry predicates over values not visible here (whether the
    // dispatcher is permitted at all for this sub-stream, whether a guest barrier made it unsafe,
    // the branch count, the presence of particular ops). At least one caller withholds the
    // dispatcher outright, so "refused loop" and "loop emulated" are not the same claim. That is as
    // far as this function can see, and it is far enough to know the consequence is not its to
    // state. Plumbing those values in purely to keep a sentence would be the wrong trade: the
    // sentence is what should go.
    //
    // (Found reviewing #2695, which fixed the same over-assertion one level up. Counting both
    // earlier drafts of this comment -- one asserting the dispatcher, one asserting outright
    // rejection, one asserting a false universal over the entries -- the shared shape is that the
    // clause gets written where the CAUSE is detected rather than where the EFFECT is decided, and
    // those are different functions. Writing a *narrower* claim is not a fix if it is still a claim
    // about the other function.)
    //
    // `caller_role` is in the message AND in the de-dup key, and both halves are load-bearing.
    // Two callers reach this function with the same `safe` set and the same program, and their
    // consequences are NOT the same:
    //
    //   emit            -- the recompile. Every loop in the shader is discarded; what the stream
    //                      does next is decided by the caller, not here (see the note above).
    //   multiwave-probe -- `compute_shader_prefers_native_multiwave`, which only tests
    //                      `loops.empty()`. Nothing is discarded and no fallback happens; the
    //                      preference analysis simply proceeds as though the shader were loop-free.
    //
    // Without the role in the KEY the probe silences the recompile, because the probe runs first
    // (`gpu_executor.cpp` calls it before the recompile). Without the role in the MESSAGE the
    // surviving line states the recompile's consequence over the probe's cause -- which reads as
    // proof that a recompile discarded the loops when no recompile has happened yet. That is
    // precisely the misreading this diagnostic exists to prevent, so it must not manufacture it.
    auto divloop_reject = [&](const char* why, uint32_t pc) -> std::vector<DivLoop> {
        if (getenv("PROSPER_DBG") && diagnostic.program_address != 0) {
            static std::mutex seen_mu;
            // Keyed by string VALUE, not by pointer: every caller passes a literal today, so
            // pointer identity would happen to work, and would break silently the first time
            // someone passes a computed name.
            static std::set<std::tuple<uint64_t, std::string, uint32_t, std::string>> seen;
            std::lock_guard<std::mutex> lk(seen_mu);
            // The key space is (programs x causes x pcs x roles) and a long session keeps loading
            // new shaders, so leave nothing unbounded even on a diagnostic path. Clearing rather
            // than refusing to insert means the reporter degrades into REPEATING lines, never into
            // dropping them -- for a diagnostic, losing tidiness beats losing findings. Measured at
            // 14 tuples on a full routed GTA V run, so the cap is not expected to be reached.
            if (seen.size() >= 4096u) seen.clear();
            if (seen.emplace(diagnostic.program_address, why, pc, caller_role).second)
                std::fprintf(stderr,
                             "[divloop-reject] program=0x%llx role=%s %s at pc=%u (%s)\n",
                             (unsigned long long)diagnostic.program_address, caller_role, why, pc,
                             std::strcmp(caller_role, "emit") == 0
                                 ? "all loops discarded; what the stream does next is the caller's"
                                 : "probe only: nothing discarded, analysis proceeds as loop-free");
        }
        return {};
    };
    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    // Pass 1: each backward s_branch / s_cbranch_execnz is a candidate back-edge.
    std::vector<bool> backedge_execnz;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        if ((in.opcode != 0x02 && in.opcode != 0x09) || in.simm16 >= 0) continue;
        if (safe.count(in.pc)) continue;
        DivLoop L;
        L.header_pc = branch_target(in);
        L.backedge_pc = in.pc;
        L.exit_pc = in.pc + in.len_dwords;
        if (L.header_pc >= L.backedge_pc) return divloop_reject("degenerate-loop-bounds", L.header_pc);
        if (L.exit_pc > end_pc) return divloop_reject("exit-past-end-of-program", L.header_pc);
        bool hdr_ok = false;
        for (const auto& h : ins) { if (h.pc == L.header_pc) { hdr_ok = true; break; } if (h.pc > L.header_pc) break; }
        if (!hdr_ok) return divloop_reject("header-pc-is-not-an-instruction", L.header_pc);
        out.push_back(L);
        backedge_execnz.push_back(in.opcode == 0x09);
    }
    if (out.empty()) return out;
    // Loops may be strictly DISJOINT (sequential) or PROPERLY NESTED (#590 — DOLL's post-process
    // kernel iterates an inner table loop inside an outer row loop). Pass 1 collected loops in
    // BACK-EDGE pc order (an inner back-edge precedes its outer's); re-sort by header so the list is
    // in emission order — emit_structured consumes loops by header pc, and emitting an outer body
    // recursively reaches the inner loop first because its header lies inside that range. Then
    // validate every pair as either disjoint-in-order or properly nested; partial overlap (an
    // unstructured region) still rejects loudly.
    {
        std::vector<size_t> order(out.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t c) { return out[a].header_pc < out[c].header_pc; });
        std::vector<DivLoop> sorted_loops; std::vector<bool> sorted_execnz;
        sorted_loops.reserve(out.size()); sorted_execnz.reserve(out.size());
        for (size_t i : order) { sorted_loops.push_back(out[i]); sorted_execnz.push_back(backedge_execnz[i]); }
        out.swap(sorted_loops); backedge_execnz.swap(sorted_execnz);
    }
    for (size_t i = 0; i < out.size(); i++)
        for (size_t j = i + 1; j < out.size(); j++) {
            const DivLoop& A = out[i]; const DivLoop& B = out[j];   // A.header_pc <= B.header_pc
            if (A.header_pc == B.header_pc) return divloop_reject("shared-header", A.header_pc); // shared header: not modeled
            const bool disjoint = B.header_pc >= A.exit_pc;
            const bool nested   = B.exit_pc <= A.backedge_pc;       // B entirely inside A's body
            if (!disjoint && !nested) return divloop_reject("partial-overlap", B.header_pc);    // partial overlap: unstructured
        }
    // Pass 2: validate each loop's interior branches and find the canonical exit + breaks. A branch
    // inside a NESTED child loop belongs to the child (which validates itself in its own pass-2
    // iteration) — skip it here, including the child's backward back-edge, which is exactly the
    // "second back-edge inside" that used to reject nesting outright.
    auto inside_nested_child = [&out](const DivLoop& L, uint32_t pc) {
        for (const auto& C : out)
            if (C.header_pc > L.header_pc && C.exit_pc <= L.backedge_pc &&
                pc >= C.header_pc && pc <= C.backedge_pc) return true;
        return false;
    };
    for (size_t li = 0; li < out.size(); li++) {
        DivLoop& L = out[li];
        const bool execnz = backedge_execnz[li];
        for (const auto& in : ins) {
            if (in.is_end || in.pc >= L.backedge_pc) break;
            if (in.pc < L.header_pc || in.fmt != Rdna2Format::SOPP) continue;
            if (inside_nested_child(L, in.pc)) continue;
            switch (in.opcode) {
                case 0x02: case 0x04: case 0x05: case 0x06: case 0x07: case 0x08: case 0x09: break;
                default: continue;   // hints
            }
            if (in.simm16 < 0) return divloop_reject("nested-backedge-in-body", in.pc);         // second back-edge inside -> nested loop
            if (safe.count(in.pc)) continue;                    // linearized (kill-mask / safe-execz)
            uint32_t tgt = branch_target(in);
            if (in.opcode == 0x02) {                            // forward s_branch: an else-arm terminator
                if (tgt > L.backedge_pc) return divloop_reject("else-arm-branch-leaves-body", in.pc); // may not leave the body
                continue;                                       // (validated by detect_forward_ifs)
            }
            if (tgt > L.exit_pc) return divloop_reject("conditional-branch-past-loop", in.pc);      // conditional jumping past the loop
            if (tgt == L.exit_pc) {                             // an exit test
                if (!L.exit_branch_pc) {                        // first one = the canonical exit
                    if (in.opcode == 0x08) {                    // execz -> EXIT
                        L.condition = DivLoop::Condition::Exec;
                    } else if (in.opcode == 0x06 && !execnz &&
                               (exact_fragment_wave_breaks ||
                                vcc_exit_is_wave_uniform(ins, in.pc))) {
                        // Fragment reduces the real VCC mask through a forced wave64 vote. Other
                        // structured stages require the compare to be proven uniform first.
                        L.condition = DivLoop::Condition::Vcc;
                    } else if ((in.opcode == 0x04 || in.opcode == 0x05) && !execnz) {
                        // A scalar SCC exit with an unconditional back-edge is the same canonical
                        // top-tested while loop accepted by CountedLoop. Keep it in this multi-loop
                        // region form when several disjoint/nested loops coexist in one shader.
                        // scc0 exits while SCC==0 (continue on set); scc1 has the opposite polarity.
                        L.condition = DivLoop::Condition::Scc;
                        L.continue_on_set = in.opcode == 0x04;
                    } else if (in.opcode == 0x06 && !execnz) {
                        // Reached the vccz arm but the compare was not proven wave-uniform.
                        return divloop_reject("vcc-exit-not-proven-wave-uniform", in.pc);
                    } else if (execnz) {
                        // A scalar/vcc exit test combined with an execnz back-edge: the bottom-tested
                        // lowering owns that back-edge, so this exit shape is not the one modeled.
                        return divloop_reject("scalar-exit-with-execnz-backedge", in.pc);
                    } else {
                        return divloop_reject("exit-branch-opcode-not-modeled", in.pc);
                    }
                    L.exit_branch_pc = in.pc;
                } else {                                        // later ones = breaks
                    if (in.opcode != 0x06 && in.opcode != 0x08) return divloop_reject("break-not-vccz-or-execz", in.pc); // vccz/execz only
                    // With an execnz back-edge, a cleared lane skips to the next header check. With
                    // an unconditional back-edge that could re-enable EXEC, only an EXECZ branch is
                    // exact: the emitter sends the cleared lane directly to the loop merge instead.
                    // A VCCZ break does not itself clear EXEC and remains unsupported for that shape.
                    if (!execnz && in.opcode != 0x08) {
                        // A VCCZ break does not clear per-lane EXEC, so the old lane-local loop
                        // approximation could not carry it to the merge. The fragment shell now
                        // reduces VCC over an enforced wave64 subgroup and can branch the complete
                        // guest wave directly at the loop latch. Other stages remain conservative.
                        if (!exact_fragment_wave_breaks || in.opcode != 0x06) return divloop_reject("vccz-break-unsupported-stage", in.pc);
                        L.direct_wave_breaks = true;
                    } else if (!execnz) {
                        L.direct_exec_breaks = true;
                    }
                    L.break_pcs.push_back(in.pc);
                }
            }
            // (tgt <= backedge_pc: an interior forward if — validated by detect_forward_ifs.)
        }
        if (!L.exit_branch_pc) {
            // Bottom-tested EXEC loop: the back-edge itself is `s_cbranch_execnz HEADER`, so the
            // complete body executes once before the first condition. Require the instruction
            // immediately preceding the branch to update EXEC; accepting a stale entry mask here
            // would turn an arbitrary backward branch into a do-while (and commonly an infinite
            // one). GTA V's nested lighting loop ends in V_CMPX; the regression uses S_MOV_B64
            // EXEC,0. Nested child branches were already validated and skipped above.
            if (!execnz) return divloop_reject("bottom-tested-needs-execnz-backedge", L.backedge_pc);
            const Rdna2Inst* condition_writer = nullptr;
            for (const auto& in : ins) {
                if (in.pc < L.header_pc) continue;
                if (in.pc >= L.backedge_pc) break;
                condition_writer = &in;
            }
            if (!condition_writer ||
                !rdna2_instruction_may_change_exec(*condition_writer)) return divloop_reject("bottom-tested-no-exec-writer", L.backedge_pc);
            L.exit_branch_pc = L.backedge_pc;
            L.condition = DivLoop::Condition::Exec;
            L.continue_on_set = true;
            L.bottom_tested = true;
        }
        if (!L.bottom_tested) {
            // A top-tested loop's canonical exit must be the FIRST branch, so its condition region
            // [header, exit_branch) is branch-free and can be emitted straight-line. A bottom-tested
            // loop instead puts its complete, already-validated structured body in this region.
            for (const auto& in : ins) {
                if (in.pc >= L.exit_branch_pc) break;
                if (in.pc < L.header_pc || in.fmt != Rdna2Format::SOPP) continue;
                if (in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03 &&
                    !safe.count(in.pc)) return divloop_reject("condition-region-not-branch-free", in.pc);
            }
        }
        // The execnz flavor's unconditional-continue lowering requires the header check to
        // immediately re-test EXEC (empty condition region) — see the shape comment.
        if (execnz && !L.bottom_tested && L.exit_branch_pc != L.header_pc) return divloop_reject("execnz-nonempty-condition-region", L.header_pc);
    }
    // A nested child must lie entirely within its parent's BODY: after the parent's canonical exit
    // test (the condition region [header, exit_branch) stays branch-free) and before its back-edge.
    for (size_t i = 0; i < out.size(); i++)
        for (size_t j = i + 1; j < out.size(); j++)
            if (out[j].exit_pc <= out[i].backedge_pc &&        // nested per the classification above
                !out[i].bottom_tested &&
                out[j].header_pc <= out[i].exit_branch_pc) return divloop_reject("child-header-inside-parent-condition-region", out[j].header_pc);
    // Pass 3: no branch from OUTSIDE a loop may target its interior (an unstructured entry edge).
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        if (in.opcode < 0x02 || in.opcode > 0x09 || in.opcode == 0x03) continue;
        uint32_t tgt = branch_target(in);
        for (const auto& L : out) {
            const bool inside_br = in.pc >= L.header_pc && in.pc <= L.backedge_pc;
            const bool inside_tgt = tgt > L.header_pc && tgt < L.exit_pc;
            if (!inside_br && inside_tgt) return divloop_reject("entry-edge-from-outside-loop", in.pc);
        }
    }
    return out;
}

// A single structured uniform IF: exactly ONE control-flow branch in the whole stream, and it is a
// FORWARD s_cbranch_scc0/scc1 (a scalar-uniform conditional — all lanes take the same path). The
// conditional block is [branch_pc+1, target_pc). Anything more (any s_branch/vcc/execz/execnz branch,
// a backward branch/loop, or a target outside the stream) is rejected → falls back to straight-line
// (which then rejects the scc branch, as before). Deliberately conservative: covers the common single
// forward uniform-if without a general structurizer.
struct ForwardIf {
    bool found = false;
    uint32_t branch_pc = 0, target_pc = 0;
    bool on_scc0 = true;   // s_cbranch_scc0/vccz/execz (branch taken == skip block when flag==0) vs scc1/vccnz
    bool on_vcc  = false;  // condition register: false = SCC, true = VCC (both are wave-uniform branches)
    bool early_out = false;// branch target was clamped to end_pc: the conditional block ENDS the shader
    // Divergent-region if (#273): the branch is a forward s_cbranch_execz over an EXEC-narrowed block
    // (v_cmpx … s_cbranch_execz — DOLL's FXAA PS). Fragment reduces EXEC across an enforced wave64;
    // other structured stages use the guarded per-invocation model. EXEC is phi'd at the merge.
    bool on_exec = false;
    // Compute only (#1554): this VCC branch was PROVED to decide identically for every wave in the
    // workgroup, so its region may contain guest barriers and nested wave votes. Left false for every
    // branch whose uniformity is not proved, which keeps the conservative per-wave treatment.
    bool uniform_workgroup = false;
    // IF/ELSE (#273): the then-arm [branch_pc+1, sb_pc) is terminated by `s_branch merge_pc` at sb_pc
    // (the instruction immediately before target_pc); the else-arm starts at target_pc and runs to
    // merge_pc (or to the enclosing region's end when merge_pc is the shared OUTER merge — DOLL's
    // color-grade if/else-if cascade, where every arm's s_branch jumps to the outermost merge).
    bool has_else = false;
    uint32_t sb_pc = 0, merge_pc = 0;
};

// MULTIPLE structured uniform IFs — the generalization of detect_forward_if to N forward
// s_cbranch_scc*/vcc* branches that form a properly-NESTED (or disjoint sequential) region tree.
// Real shaders routinely carry several sequential uniform ifs (DOLL's color-grade PS: two
// `s_cbranch_vccz` blocks) or a nested guard (`vccz L1; …; vccz L2` with L2 < L1). Same per-branch
// validity rules as detect_forward_if (forward target inside the stream, or a VERIFIED early-out
// clamped to s_endpgm; scc branches in `skip` are alpha-test kill-masks handled by linearization);
// plus a structural check: blocks must nest or be disjoint — a partial overlap (branch into the
// middle of another block) is not an if-tree and rejects. Returns branches in pc order; empty =
// no/unsupported control flow (the caller falls back to straight-line, which rejects loudly at the
// branch). Compute execz uses a subgroup vote; execnz and unclaimed s_branch still reject wholesale.
// CONFIDENCE: HIGH on the structure (guarded by the phi machinery shared with the single-if path).
inline std::vector<ForwardIf> detect_forward_ifs(const std::vector<Rdna2Inst>& ins, bool allow_vcc,
                                          const uint32_t* code, size_t dwords,
                                          const std::unordered_set<uint32_t>* skip = nullptr,
                                          const std::vector<DivLoop>* loops = nullptr,
                                          bool* rejected = nullptr,
                                          bool compute_wave_branches = false,
                                          RecompileDiagnosticContext diagnostic = {}) {
    std::vector<ForwardIf> out;
    if (rejected) *rejected = false;
    auto reject = [&]() -> std::vector<ForwardIf> { if (rejected) *rejected = true; return {}; };
    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    // Loop-claimed pcs (#273): back-edges + canonical exits are consumed by the loop emitter; a break
    // lowers as a forward if over the remainder of its loop's body (target clamped to the back-edge).
    auto loop_backedge = [&](uint32_t pc) { if (loops) for (const auto& L : *loops) if (L.backedge_pc == pc) return true; return false; };
    auto loop_exit     = [&](uint32_t pc) { if (loops) for (const auto& L : *loops) if (L.exit_branch_pc == pc) return true; return false; };
    auto loop_break    = [&](uint32_t pc) -> const DivLoop* {
        if (loops) for (const auto& L : *loops)
            for (uint32_t bp : L.break_pcs) if (bp == pc) return &L;
        return nullptr;
    };
    // Pass 1: collect the conditional branches (else-arm s_branch terminators are claimed in pass 2).
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        // Set when a compute vccz/vccnz if is accepted for exact guest-wave control flow. Its region
        // must remain free of operations that require every workgroup invocation to reach one common
        // synchronized phase (checked after the target/merge is known).
        bool compute_uniform_vcc = false;
        switch (in.opcode) {
            case 0x02: continue;                                     // s_branch: validated in pass 2
            case 0x09:                                               // execnz: only as a loop back-edge
                if (loop_backedge(in.pc)) continue;
                log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                         "unclaimed execnz pc=%u", in.pc);
                return reject();
            case 0x08:                                               // execz: safe-linearized predication
                if (skip && skip->count(in.pc)) continue;            // branch -> emit_alu no-ops it; else a
                if (loop_exit(in.pc)) continue;                      // loop exit test: the loop emitter's
                if (!allow_vcc && !compute_wave_branches) {
                    log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                             "unsupported execz pc=%u", in.pc);
                    return reject();
                }
                break;                                               // compute lowers the wave-empty test with
                                                                     // OpGroupNonUniformAny(EXEC)
            case 0x06: case 0x07:                                    // vccz / vccnz
                if (loop_exit(in.pc)) continue;                      // canonical loop condition: loop emitter owns it
                if (!allow_vcc) {
                    // Compute VCC is a per-lane bool in the SPIR-V shell. The scalar branch consumes
                    // the architectural whole-wave VCCZ/VCCNZ flag. Accept it for the exact guest-wave
                    // CFG dispatcher; emit_body must never lower it through a native subgroup vote.
                    if (!compute_wave_branches) {
                        log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                                 "unsupported vcc branch pc=%u", in.pc);
                        return reject();
                    }
                    compute_uniform_vcc = true;
                }
                break;
            case 0x04: case 0x05:                                    // scc0 / scc1
                if (loop_exit(in.pc)) continue;                      // scalar loop condition: loop emitter owns it
                if (skip && skip->count(in.pc)) continue;            // kill-mask branch: linearized, not an if
                break;
            default: continue;                                       // hints
        }
        // A loop BREAK (validated vccz/execz -> exit_pc inside an execnz-back-edge loop): a plain
        // forward if skipping the REST of the body — the exec-governed continue then exits the lane.
        // A compute-uniform-accepted vcc branch can never be a break (compute loops require the
        // s_branch back-edge Vcc shape, which has no breaks) — reject the combination defensively.
        if (compute_uniform_vcc && loop_break(in.pc)) {
            log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                     "uniform vcc loop break pc=%u", in.pc);
            return reject();
        }
        if (const DivLoop* L = loop_break(in.pc)) {
            ForwardIf F;
            F.found = true; F.branch_pc = in.pc; F.target_pc = L->backedge_pc;
            F.on_scc0 = true;                                        // vccz/execz: skip block when flag==0
            F.on_vcc  = (in.opcode == 0x06);
            F.on_exec = (in.opcode == 0x08);
            out.push_back(F);
            continue;
        }
        uint32_t tgt = branch_target(in);
        bool early = false;
        if (tgt > end_pc && tgt != UINT32_MAX) {                     // possible early-out past s_endpgm:
            if (!code || tgt >= dwords) {
                log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                         "target out of range pc=%u target=%u", in.pc, tgt);
                return reject();                                     // verify the target terminates (see
            }
            std::vector<Rdna2Inst> tail;                             // detect_forward_if for the rationale)
            rdna2_walk(code + tgt, dwords - tgt, tail);
            bool ends_immediately = false;
            for (const auto& ti : tail) {
                if (ti.is_end) { ends_immediately = true; break; }
                if (ti.fmt == Rdna2Format::SOPP && ti.opcode == 0x00) continue;
                break;
            }
            if (!ends_immediately) {
                log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                         "nonterminal early target pc=%u target=%u", in.pc, tgt);
                return reject();
            }
            tgt = end_pc; early = true;
        }
        if (tgt <= in.pc || tgt > end_pc) {
            log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                     "nonforward target pc=%u target=%u", in.pc, tgt);
            return reject();                                         // must be forward, within the stream
        }
        ForwardIf F;
        F.found = true; F.branch_pc = in.pc; F.target_pc = tgt; F.early_out = (early || tgt == end_pc);
        F.on_scc0 = (in.opcode == 0x04 || in.opcode == 0x06 || in.opcode == 0x08);
        F.on_vcc  = (in.opcode == 0x06 || in.opcode == 0x07);
        F.on_exec = (in.opcode == 0x08);
        // IF/ELSE detection: the instruction immediately preceding target_pc is a FORWARD s_branch —
        // the then-arm's terminator jumping to the merge; the else-arm is [target_pc, merge). (Skip
        // for early-outs: the arm ends the shader.)
        if (!early && tgt != end_pc) {
            for (const auto& sb : ins) {
                if (sb.pc <= in.pc || sb.pc >= tgt) continue;
                if (sb.fmt != Rdna2Format::SOPP || sb.opcode != 0x02) continue;
                if (sb.pc + sb.len_dwords != tgt) continue;          // must be the arm's LAST instruction
                if (loop_backedge(sb.pc)) continue;                  // a loop's back-edge, not an else-jump
                if (sb.simm16 <= 0) {
                    log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                             "backward else pc=%u branch=%u target=%u",
                                             in.pc, sb.pc, tgt);
                    return reject();                                 // backward else-jump: a loop, not an if
                }
                uint32_t lm = branch_target(sb);
                if (lm <= tgt || lm > end_pc) {
                    log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                             "invalid else merge pc=%u branch=%u target=%u merge=%u",
                                             in.pc, sb.pc, tgt, lm);
                    return reject();                                 // merge must be forward, in-stream
                }
                F.has_else = true; F.sb_pc = sb.pc; F.merge_pc = lm;
                break;
            }
        }
        if (!allow_vcc && compute_wave_branches && F.on_exec) {
            // A compute execz condition is subgroup-uniform, so live scalar writes and LDS are safe
            // inside the structured arm. A workgroup barrier is not: different waves may take
            // different arms. Scalar writes into VCC_LO/HI also have an implicit mask-domain side
            // effect which the scalar scratch model cannot reconstruct; accept those only when the
            // mask is provably overwritten before any post-merge implicit read. The captured UE4
            // kernel writes VCC_LO as an integer scratch in the arm and overwrites VCC with a VOPC at
            // the merge, which this deliberately narrow proof accepts.
            const uint32_t region_end = F.has_else ? F.merge_pc : F.target_pc;
            for (const auto& r : ins) {
                if (r.is_end || r.pc >= region_end) break;
                if (r.pc <= in.pc) continue;
                if (r.fmt == Rdna2Format::SOPP && r.opcode == 0x0a) {
                    log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                             "execz pc=%u contains barrier pc=%u", in.pc, r.pc);
                    return reject();
                }

                uint32_t scalar_width = 0;
                if (r.fmt == Rdna2Format::SOP1) scalar_width = r.opcode == 0x04 ? 2u : 1u;
                else if (r.fmt == Rdna2Format::SOP2) scalar_width = 1;
                else if (r.fmt == Rdna2Format::SOPK && sopk_writes_scalar_data(r.opcode)) scalar_width = 1;
                else if (r.fmt == Rdna2Format::SMEM) {
                    switch (r.opcode) {
                        case 0x0: case 0x8: scalar_width = 1; break;
                        case 0x1: case 0x9: scalar_width = 2; break;
                        case 0x2: case 0xA: scalar_width = 4; break;
                        case 0x3: case 0xB: scalar_width = 8; break;
                        case 0x4: case 0xC: scalar_width = 16; break;
                        default: break;
                    }
                }
                if (!scalar_width ||
                    (r.dst.kind != OperandKind::SGPR && r.dst.kind != OperandKind::Special)) continue;
                for (int vcc_half = 106; vcc_half <= 107; ++vcc_half) {
                    if (vcc_half < r.dst.value || vcc_half >= r.dst.value + (int)scalar_width) continue;
                    // PROSPER_VCC_SCALAR_DATA_MERGE=1 widens the proof from "not read" to "not read
                    // as a lane mask". The guard this loop implements is about the MASK domain --
                    // its own comment says so -- but the test it applies is total liveness, so a
                    // half that survives the merge only to be read as a 32-bit scalar dword is
                    // rejected even though the scalar model reproduces that dword exactly.
                    //
                    // Opt-in rather than default because the same physical pair being a mask on one
                    // predecessor and data on another has no runtime type tag, and this file already
                    // rejects that join deliberately. Keeping it behind a switch also makes the A/B
                    // have a lever that can be shown to have moved, which a byte-identical module
                    // has already cost this investigation once.
                    static const bool relax_vcc_scalar_data =
                        std::getenv("PROSPER_VCC_SCALAR_DATA_MERGE") != nullptr;
                    const ScalarMergeProof proof = relax_vcc_scalar_data
                        ? ScalarMergeProof::MaskDomainOnly : ScalarMergeProof::AnyRead;
                    ScalarMergeBlocker blocker;
                    if (!sgpr_dead_at_merge(ins, region_end, vcc_half, proof, &blocker)) {
                        log_recompile_diagnostic(
                            diagnostic, "compute-struct-reject", "route-decline",
                            "execz pc=%u scalar write pc=%u leaves vcc-half s%d live at merge pc=%u "
                            "blocked-by pc=%u kind=%s",
                            in.pc, r.pc, vcc_half, region_end, blocker.pc, blocker.kind);
                        return reject();
                    }
                }
            }
        }
        // A branch whose mask is built only from launch data over a full EXEC decides identically for
        // every wave of the workgroup (#1554), so its region may synchronize. Prove it before the
        // per-wave restriction below applies.
        if (compute_uniform_vcc && vcc_branch_is_workgroup_uniform(ins, in.pc)) {
            F.uniform_workgroup = true;
            compute_uniform_vcc = false;
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr, "[compute-cfg] workgroup-uniform vcc branch pc=%u\n", in.pc);
        }
        if (compute_uniform_vcc) {
            // The exact branch decision is per-WAVE, so different waves in one workgroup may take
            // different arms. Ordinary LDS loads/stores/atomics are valid under that control flow:
            // they are per-invocation memory effects and impose no reconvergence requirement. A
            // guest barrier is not valid there, and synthesized wave collectives need the CFG
            // dispatcher's common phase rather than the compact structured emitter. Keep those
            // synchronized operations fail-visible here; raw LDS memory effects can remain in-arm.
            const uint32_t region_end = F.has_else ? F.merge_pc : F.target_pc;
            for (const auto& r : ins) {
                if (r.is_end || r.pc >= region_end) break;
                if (r.pc <= in.pc) continue;
                const bool ds_wave_collective = r.fmt == Rdna2Format::DS &&
                    (r.opcode == 0x35 || r.opcode == 0x3d || r.opcode == 0x3e ||
                     r.opcode == kDsOpcodeBpermuteB32);
                if ((r.fmt == Rdna2Format::SOPP && r.opcode == 0x0a) ||
                    ds_wave_collective ||
                    (r.fmt == Rdna2Format::VOP3 &&
                     (r.opcode == 0x365 || r.opcode == 0x366))) {
                    log_recompile_diagnostic(
                        diagnostic, "compute-struct-reject", "route-decline",
                        "vcc branch pc=%u contains synchronized op pc=%u fmt=%d op=0x%x",
                        in.pc, r.pc, static_cast<int>(r.fmt), r.opcode);
                    return reject();
                }
            }
        }
        out.push_back(F);
    }
    // Pass 2: every s_branch in the stream must be a claimed else-arm terminator or a loop back-edge —
    // any other unconditional branch is control flow this structurizer doesn't model.
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x02) continue;
        if (loop_backedge(in.pc)) continue;
        bool claimed = false;
        for (const auto& F : out) if (F.has_else && F.sb_pc == in.pc) { claimed = true; break; }
        if (!claimed) {
            log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                     "unclaimed s_branch pc=%u target=%u",
                                     in.pc, branch_target(in));
            return reject();
        }
    }
    // Structural nesting check (branches are already in pc order): each region must lie entirely
    // inside the innermost still-open region, or start after it closed. A region spans to its merge
    // (if/else) or target (plain if); a LOOP is an opaque region [header_pc, exit_pc) that must nest
    // strictly. EXCEPTION (the shared-outer-merge cascade): an if/else whose merge_pc escapes the
    // innermost open region is accepted here and re-validated during emission (emit_structured
    // requires the escaping merge to equal the enclosing merge chain's continuation, i.e. no
    // instructions are skipped) — a violation rejects the shader there, fail-visible.
    std::vector<uint32_t> open;
    {
        size_t fi = 0, li = 0;
        const size_t nl = loops ? loops->size() : 0;
        while (fi < out.size() || li < nl) {
            const bool take_loop = li < nl &&
                (fi >= out.size() || (*loops)[li].header_pc < out[fi].branch_pc);
            const uint32_t start = take_loop ? (*loops)[li].header_pc : out[fi].branch_pc;
            uint32_t span_end    = take_loop ? (*loops)[li].exit_pc
                                 : (out[fi].has_else ? out[fi].merge_pc : out[fi].target_pc);
            const bool clampable = !take_loop && out[fi].has_else;
            while (!open.empty() && open.back() <= start) open.pop_back();
            if (!open.empty() && span_end > open.back()) {
                if (!clampable) {
                    log_recompile_diagnostic(
                        diagnostic, "compute-struct-reject", "route-decline",
                        "overlapping region start=%u end=%u parent-end=%u",
                        start, span_end, open.back());
                    return reject();                     // plain if / loop escaping its region: not a tree
                }
                span_end = open.back();                  // cascade if/else: clamped to the enclosing
            }                                            // region for nesting purposes (see above)
            open.push_back(span_end);
            if (take_loop) li++; else fi++;
        }
    }
    return out;
}
inline uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source) {
    switch (in.fmt) {
        case Rdna2Format::SOP1:
            if (in.opcode == kSop1OpcodeGetpcB64) return UINT32_MAX; // s_getpc has no source
            if (in.opcode == kSop1OpcodeBcnt1I32B64 ||
                in.opcode == kSop1OpcodeFf1I32B64 ||
                in.opcode == kSop1OpcodeFlbitI32B64 ||
                in.opcode == 0x18 || in.opcode == kSop1OpcodeQuadmaskB64)
                return 2;                    // bcnt/ff1/flbit consume a B64 pair
            return scalar_write_width(in) == 1 ? 1u : 2u;
        case Rdna2Format::SOP2:
            if (in.opcode == 0x1f || in.opcode == 0x21)
                return source == 1 ? 1u : 2u; // B64 shifts use a B32 shift count
            if (in.opcode == kSop2OpcodeBfmB64) return 1;  // BFM width and offset are B32
            if (in.opcode == kSop2OpcodeBfeU64)
                return source == 1 ? 1u : 2u;
            return scalar_write_width(in) == 1 ? 1u : 2u;
        case Rdna2Format::SOPC:
            return in.opcode == 0x12 || in.opcode == 0x13 ? 2u : 1u;
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
            return 1;
        case Rdna2Format::VOPC: {
            const uint32_t effective = vopc_is_cmpx(in.opcode)
                ? in.opcode - 0x10u : in.opcode;
            const bool integer64 =
                (effective >= 0xa1u && effective <= 0xa6u) ||
                (effective >= 0xe1u && effective <= 0xe6u);
            return integer64 ? 2u : 1u;
        }
        case Rdna2Format::VOP3:
            if (in.opcode == 0x360 || in.opcode == 0x361 ||
                in.opcode == 0x365 || in.opcode == 0x366 || in.opcode == 0x11f)
                return source < 2 ? 1u : UINT32_MAX;
            if (in.opcode == kVop3OpcodeLshlrevB64 ||
                in.opcode == kVop3OpcodeLshrrevB64)
                return source == 0 ? 1u : (source == 1 ? 2u : UINT32_MAX);
            if (in.opcode == 0x101)
                return source < 2 ? 1u : 2u; // cndmask data is B32; condition is a pair
            if (in.opcode == 0x141 || in.opcode == 0x143 || in.opcode == 0x347 ||
                in.opcode == 0x36f || in.opcode == kVop3OpcodeAdd3U32 ||
                in.opcode == kVop3OpcodeAndOrB32)
                return 1;
            return 2;
        default:
            return 0;
    }
}

// Scalar instructions whose encoded destination is also an implicit source. These reads must be
// observed before writer transfer functions invalidate the old lifetime. SOPK has no decoded src
// operands at all; conditional moves preserve the old destination on one SCC outcome, comparisons
// consume SDST without writing it, and ADDK/MULK are ordinary read-modify-write operations. SOP1
// conditional moves and bitset operations have an explicit source too, but it is the replacement
// value/bit index rather than the old destination.
inline uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in) {
    if (in.dst.kind != OperandKind::SGPR) return 0;
    if (in.fmt == Rdna2Format::SOPK) {
        if (in.opcode == 0x02 ||                         // s_cmovk_i32
            (in.opcode >= 0x03 && in.opcode <= 0x10))   // s_cmpk_*, s_addk, s_mulk
            return 1;
        return 0;
    }
    if (in.fmt != Rdna2Format::SOP1) return 0;
    switch (in.opcode) {
        case kSop1OpcodeCmovB32: return 1;
        case kSop1OpcodeCmovB64: return 2;
        case kSop1OpcodeBitset0B32: return 1;
        case 0x1c: return 2; // s_bitset0_b64
        case kSop1OpcodeBitset1B32: return 1;
        case 0x1e: return 2; // s_bitset1_b64
        default: return 0;
    }
}

// ---------------------------------------------------------------------------------------------
// Workgroup-uniform wave branch (#1554).
//
// A compute `s_cbranch_vccz`/`vccnz` decides per WAVE, so two waves of one workgroup may take
// different arms. A guest `S_BARRIER` inside such an arm violates Vulkan's uniform-participation
// requirement for OpControlBarrier, so detect_forward_ifs rejects that combination.
//
// Real kernels nevertheless build some of these masks entirely out of launch state:
//
//     s_load_dwordx4        s[8:11], s[12:13], 0x20    ; s12:s13 is entry data
//     s_buffer_load_dwordx2 s[16:17], s[8:11], 0xc
//     s_add_i32             vcc_lo, s17, -1            ; VCC_LO used as scalar scratch
//     s_cmp_gt_u32          s16, 0
//     s_cselect_b64         s[10:11], exec, 0
//     s_cmp_eq_u32          s14, vcc_lo
//     s_cselect_b64         vcc, exec, 0
//     s_and_b64             vcc, vcc, s[10:11]
//     s_cbranch_vccz        END
//
// With EXEC full, VCC is nonzero exactly when the scalar predicate holds, and that predicate reads
// only compute launch SGPRs. recompile_compute seeds those from push-constant user data and from
// WorkGroupId system SGPRs (see the entry seeding in recompile_compute), both of which are identical
// for every wave of a workgroup — so the branch is workgroup-uniform and its arm may synchronize.
//
// Two obligations are proved independently and mechanically:
//
//   (1) EXEC is full at the branch on EVERY path. Without it, VCCZ is also true for a wave whose
//       EXEC happens to be empty, which is a genuinely per-wave property. A forward must-dataflow
//       proves this, so a loop that restores `s_mov_b64 exec, -1` before its back-edge still
//       qualifies while an arm that narrows EXEC does not.
//   (2) VCC is `SCC ? EXEC : 0`, optionally combined by s_and_b64/s_or_b64; every contributing SCC
//       comes from a scalar compare, and every scalar input traces to launch data.
//
// Everything not proved stays rejected: EXEC or SCC read as data, V_READFIRSTLANE/V_READLANE,
// VOPC-written masks, SGPRs above the launch range, and any definition reaching the branch from
// outside its own straight-line region unless it provably still holds its entry value.
//
// CONFIDENCE: HIGH — the accepted provenance set is closed and every rejection is fail-visible.

// Forward "must" dataflow over the decoded stream. `fact[i]` is true only when the tracked property
// holds at entry to ins[i] along EVERY path that reaches it. `gen` re-establishes the property,
// `kill` destroys it, and anything else passes it through. Optimistic initialization followed by
// iteration to a fixpoint is the standard greatest-fixpoint formulation, which is what makes the
// result sound across back-edges. Returns an empty vector when the stream contains an edge this walk
// cannot resolve, so every caller treats "unanalyzable" as "not proved".
template <typename Gen, typename Kill>
std::vector<uint8_t> must_fact_at(const std::vector<Rdna2Inst>& ins, Gen&& gen, Kill&& kill,
                                  bool entry_holds) {
    if (ins.empty()) return {};
    std::unordered_map<uint32_t, size_t> index_of;
    for (size_t i = 0; i < ins.size(); ++i) index_of[ins[i].pc] = i;

    auto is_branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
               in.opcode != 0x03;
    };

    std::vector<std::vector<size_t>> succ(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) continue;
        if (is_branch(in)) {
            const uint32_t target = branch_target(in);
            auto found = index_of.find(target);
            if (found != index_of.end()) succ[i].push_back(found->second);
            else if (target <= ins.back().pc) return {};   // an edge this walk cannot resolve
            // A target past the decoded stream is an early-out that terminates the wave; it
            // contributes no successor inside this function.
        }
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x02) continue;   // s_branch: no fallthrough
        if (i + 1 < ins.size()) succ[i].push_back(i + 1);
    }

    std::vector<uint8_t> fact(ins.size(), 1);
    if (!entry_holds) fact[0] = 0;
    // Each round can only turn a 1 into a 0, so the fixpoint is reached in at most size()+1 rounds.
    for (size_t round = 0; round <= ins.size() + 1; ++round) {
        std::vector<uint8_t> next(ins.size(), 1);
        if (!entry_holds) next[0] = 0;
        for (size_t i = 0; i < ins.size(); ++i) {
            uint8_t out = fact[i];
            if (gen(ins[i])) out = 1;
            else if (kill(ins[i])) out = 0;
            if (out) continue;
            for (size_t s : succ[i]) next[s] = 0;
        }
        if (next == fact) return fact;
        fact.swap(next);
    }
    return fact;
}

// `s_mov_b64 exec, -1` is the only accepted full-mask definition. A 32-bit literal written to a
// 64-bit scalar destination is zero-extended, so it activates only the low half and is NOT a full
// wave64 mask; the inline -1 constant is sign-extended and is.
inline bool exec_write_sets_full_mask(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP1 || in.opcode != 0x04 || in.n_src != 1) return false;
    if ((in.dst.kind != OperandKind::SGPR && in.dst.kind != OperandKind::Special) ||
        in.dst.value != 126)
        return false;
    return in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1;
}

inline bool vcc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc) {
    size_t branch_index = ins.size();
    for (size_t i = 0; i < ins.size(); ++i)
        if (ins[i].pc == branch_pc) { branch_index = i; break; }
    if (branch_index == ins.size() || branch_index == 0) return false;

    // Obligation (1): EXEC is full at the branch along every path.
    const std::vector<uint8_t> exec_full = must_fact_at(
        ins, exec_write_sets_full_mask,
        [](const Rdna2Inst& in) { return rdna2_instruction_may_change_exec(in); }, true);
    if (exec_full.empty() || !exec_full[branch_index]) return false;

    auto is_branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
               in.opcode != 0x03;
    };
    std::set<uint32_t> targets;
    for (const auto& in : ins)
        if (is_branch(in)) targets.insert(branch_target(in));

    // The straight-line region ending at the branch. Every definition consumed from inside it has a
    // unique reaching definition, which is what lets the backwards slice below be a textual walk.
    size_t block_start = branch_index;
    while (block_start > 0) {
        if (targets.count(ins[block_start].pc)) break;      // another edge enters here
        const Rdna2Inst& prev = ins[block_start - 1];
        if (prev.is_end || is_branch(prev)) break;
        --block_start;
    }

    auto writes_reg = [](const Rdna2Inst& in, int reg, uint32_t words) {
        bool hit = false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            if (reg < base + static_cast<int>(width) && base < reg + static_cast<int>(words))
                hit = true;
        });
        // A VOPC without an explicit scalar destination writes architectural VCC implicitly; the
        // writer inventory only reports the explicit SGPR form.
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            in.dst.kind == OperandKind::Special && in.dst.value == 106 &&
            reg < 108 && 106 < reg + static_cast<int>(words))
            hit = true;
        return hit;
    };

    auto last_writer = [&](int reg, uint32_t words, size_t before) -> size_t {
        for (size_t i = before; i-- > block_start;)
            if (writes_reg(ins[i], reg, words)) return i;
        return ins.size();
    };

    // A register with no definition inside the region is accepted only when it provably still holds
    // its wave-entry value at the branch. Entry SGPRs are launch data, hence workgroup-uniform.
    std::map<int, bool> entry_value_cache;
    auto still_entry_value = [&](int reg) {
        auto cached = entry_value_cache.find(reg);
        if (cached != entry_value_cache.end()) return cached->second;
        const std::vector<uint8_t> fact = must_fact_at(
            ins, [](const Rdna2Inst&) { return false; },
            [&](const Rdna2Inst& in) { return writes_reg(in, reg, 1); }, true);
        const bool proved = !fact.empty() && fact[branch_index] != 0;
        entry_value_cache[reg] = proved;
        return proved;
    };

    std::function<bool(int, size_t, uint32_t)> uniform_scalar;
    std::function<bool(size_t, uint32_t)> uniform_scc;
    std::function<bool(int, size_t, uint32_t)> uniform_mask;

    auto uniform_operand = [&](const Operand& op, uint32_t words, size_t before,
                               uint32_t depth) -> bool {
        switch (op.kind) {
            case OperandKind::InlineInt:
            case OperandKind::InlineFloat:
            case OperandKind::Literal:
                return true;
            case OperandKind::Special:
                if (op.value == 125) return true;                       // SGPR_NULL
                if (op.value == 126 || op.value == 127 || op.value == 253)
                    return false;                                       // EXEC and SCC are not data
                [[fallthrough]];
            case OperandKind::SGPR:
                for (uint32_t word = 0; word < words; ++word)
                    if (!uniform_scalar(op.value + static_cast<int>(word), before, depth + 1))
                        return false;
                return true;
            default:
                return false;                                           // VGPR and everything else
        }
    };

    uniform_scalar = [&](int reg, size_t before, uint32_t depth) -> bool {
        if (depth > 64) return false;
        const size_t w = last_writer(reg, 1, before);
        if (w == ins.size()) return reg >= 0 && reg <= 105 && still_entry_value(reg);
        const Rdna2Inst& in = ins[w];
        switch (in.fmt) {
            case Rdna2Format::SOP1:
                if (in.opcode == 0x1f) return true;                     // s_getpc_b64
                if (rdna2_instruction_may_change_exec(in) || in.n_src != 1) return false;
                return uniform_operand(in.src[0], scalar_write_width(in) == 2 ? 2u : 1u, w, depth);
            case Rdna2Format::SOP2: {
                // Carry and cselect forms consume SCC as well as their decoded operands; the mask
                // slice models the cselect shape explicitly and nothing else may use them as data.
                if (in.opcode == 0x04 || in.opcode == 0x05 || in.opcode == 0x0a ||
                    in.opcode == 0x0b)
                    return false;
                const uint32_t words = scalar_write_width(in) == 2 ? 2u : 1u;
                return in.n_src == 2 && uniform_operand(in.src[0], words, w, depth) &&
                       uniform_operand(in.src[1], words, w, depth);
            }
            case Rdna2Format::SMEM: {
                // Scalar-buffer loads use a four-dword descriptor; scalar-memory loads a base pair.
                if (!scalar_write_width(in) || in.n_src < 1) return false;
                const uint32_t base_words = in.opcode >= 0x8 ? 4u : 2u;
                if (!uniform_operand(in.src[0], base_words, w, depth)) return false;
                return in.n_src < 2 || uniform_operand(in.src[1], 1, w, depth);
            }
            case Rdna2Format::SOPK:
                return in.opcode == 0x00;                               // s_movk_i32 is a literal
            default:
                return false;         // VOPC masks, v_readfirstlane, v_readlane, everything else
        }
    };

    uniform_scc = [&](size_t before, uint32_t depth) -> bool {
        if (depth > 64) return false;
        for (size_t i = before; i-- > block_start;) {
            const Rdna2Inst& in = ins[i];
            if (sopp_is_noop(in)) continue;
            if (in.fmt == Rdna2Format::SOPC) {
                if (in.n_src == 0) return false;
                const uint32_t words = in.opcode == 0x12 || in.opcode == 0x13 ? 2u : 1u;
                for (uint32_t s = 0; s < in.n_src; ++s)
                    if (!uniform_operand(in.src[s], words, i, depth)) return false;
                return true;
            }
            // Any scalar ALU or SOPK instruction may write SCC. Stop rather than accidentally
            // consuming an older compare through an unmodeled SCC writer.
            if (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                in.fmt == Rdna2Format::SOPK)
                return false;
        }
        return false;                 // SCC entering the region from outside is not proved
    };

    uniform_mask = [&](int reg, size_t before, uint32_t depth) -> bool {
        if (depth > 64) return false;
        const size_t w = last_writer(reg, 2, before);
        if (w == ins.size()) return false;      // a mask entering from outside is not proved
        const Rdna2Inst& in = ins[w];
        if (in.fmt != Rdna2Format::SOP2 || in.dst.value != reg || scalar_write_width(in) != 2 ||
            in.n_src != 2)
            return false;
        auto is_exec = [](const Operand& op) {
            return (op.kind == OperandKind::SGPR || op.kind == OperandKind::Special) &&
                   op.value == 126;
        };
        auto is_zero = [](const Operand& op) {
            return op.kind == OperandKind::InlineInt && op.value == 0;
        };
        if (in.opcode == 0x0b) {                // s_cselect_b64 dst, EXEC, 0  (either polarity)
            const bool shape = (is_exec(in.src[0]) && is_zero(in.src[1])) ||
                               (is_zero(in.src[0]) && is_exec(in.src[1]));
            return shape && uniform_scc(w, depth + 1);
        }
        if (in.opcode == 0x0f || in.opcode == 0x11) {          // s_and_b64 / s_or_b64
            for (uint32_t s = 0; s < 2; ++s) {
                const Operand& op = in.src[s];
                if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) return false;
                if (op.value == 126 || op.value == 127) return false;   // raw EXEC is per-wave
                if (!uniform_mask(op.value, w, depth + 1)) return false;
            }
            return true;
        }
        return false;
    };

    // Obligation (2): architectural VCC at the branch is a select over workgroup-uniform SCCs.
    return uniform_mask(106, branch_index, 0);
}

// PC-relative EMBEDDED-TABLE detection (#273). Compilers put small constant lookup tables (a 4x4
// ordered-dither matrix, etc.) directly after the shader code and address them with an inline-built
// V#: `s_getpc_b64 s[a:a+1]; s_add_u32 sa, <off>, sa; s_addc_u32 sa+1, 0, sa+1; s_mov_b32 sa+2,
// <num_records>; s_mov_b32 sa+3, <cfg>; buffer_load_dwordx4 v, v, s[a:a+3], 0 offen`. The table
// BYTES are inside the code blob we are recompiling, so the load can be folded to a compile-time
// constant lookup. This pass does forward constant propagation over the LINEAR stream (facts are
// killed on any other write; a fact is only used when no branch target enters between its def and
// the load) and returns: mubuf pc -> the table's dwords copied out of the blob.
// CONFIDENCE: MED-HIGH — exact for the compiler idiom (verified against DOLL's dither PS); every
// guard failure falls back to the old reject.
struct PcrelTables {
    std::unordered_map<uint32_t, std::vector<uint32_t>> mubuf;
    std::unordered_map<uint32_t, std::vector<uint32_t>> smem;
};

inline PcrelTables detect_pcrel_tables(
        const std::vector<Rdna2Inst>& ins, const uint32_t* code, size_t dwords,
        size_t* required_dwords = nullptr) {
    PcrelTables out;
    std::unordered_map<int, uint64_t> pcoff;   // reg -> byte offset into the code blob (pair LO half)
    std::unordered_set<int> pchi;              // regs holding the matching HI half
    std::unordered_map<int, uint32_t> kconst;  // reg -> compile-time constant (s_mov_b32 literal/inline)
    std::unordered_map<int, uint32_t> fact_pc; // reg -> pc where its current fact was established
    std::vector<uint32_t> br_targets;          // all branch targets (for the entry-soundness check)
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) &&
            in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03)
            br_targets.push_back(in.pc + in.len_dwords + (uint32_t)(int32_t)in.simm16);
    }
    auto kill = [&](int r) { pcoff.erase(r); pchi.erase(r); kconst.erase(r); fact_pc.erase(r); };
    auto imm_of = [&](const Operand& o, const Rdna2Inst& in, uint32_t* v) -> bool {
        if (o.kind == OperandKind::Literal)   { *v = in.literal; return true; }
        if (o.kind == OperandKind::InlineInt) { *v = (uint32_t)o.value; return true; }
        if (o.kind == OperandKind::SGPR)      { auto it = kconst.find(o.value);
                                                if (it != kconst.end()) { *v = it->second; return true; } }
        return false;
    };
    for (const auto& in : ins) {
        if (in.is_end) break;
        switch (in.fmt) {
            case Rdna2Format::SOP1:
                kill(in.dst.value);
                if (in.opcode == 0x1f) {                     // s_getpc_b64: pair = address of NEXT inst
                    kill(in.dst.value + 1);
                    pcoff[in.dst.value] = (uint64_t)(in.pc + in.len_dwords) * 4u;
                    pchi.insert(in.dst.value + 1);
                    fact_pc[in.dst.value] = in.pc; fact_pc[in.dst.value + 1] = in.pc;
                } else if (in.opcode == 0x03) {              // s_mov_b32 <const>
                    uint32_t v;
                    if (imm_of(in.src[0], in, &v)) { kconst[in.dst.value] = v; fact_pc[in.dst.value] = in.pc; }
                } else if (in.opcode == 0x04) {              // s_mov_b64 (pair write)
                    kill(in.dst.value + 1);
                }
                break;
            case Rdna2Format::SOP2: {
                int d = in.dst.value;
                if (in.opcode == 0x00) {                     // s_add_u32: pcrel-lo + immediate
                    uint32_t imm; uint64_t base; bool got = false;
                    if (in.src[0].kind == OperandKind::SGPR && pcoff.count(in.src[0].value) &&
                        imm_of(in.src[1], in, &imm)) { base = pcoff[in.src[0].value]; got = true; }
                    else if (in.src[1].kind == OperandKind::SGPR && pcoff.count(in.src[1].value) &&
                             imm_of(in.src[0], in, &imm)) { base = pcoff[in.src[1].value]; got = true; }
                    kill(d);
                    if (got) { pcoff[d] = base + imm; fact_pc[d] = in.pc; }
                } else if (in.opcode == 0x04) {              // s_addc_u32: pcrel-hi + 0 (+carry; a real
                    // carry needs the lo word to wrap over a <4 KB offset — not a shader-blob shape)
                    uint32_t imm;
                    bool hi_ok = (in.src[0].kind == OperandKind::SGPR && pchi.count(in.src[0].value) &&
                                  imm_of(in.src[1], in, &imm) && imm == 0) ||
                                 (in.src[1].kind == OperandKind::SGPR && pchi.count(in.src[1].value) &&
                                  imm_of(in.src[0], in, &imm) && imm == 0);
                    kill(d);
                    if (hi_ok) { pchi.insert(d); fact_pc[d] = in.pc; }
                } else {
                    kill(d);
                    if (in.opcode == 0x29) kill(d + 1);      // s_bfe_u64 writes a pair
                }
                break;
            }
            case Rdna2Format::SOPK:
                // s_movk_i32 is also a common way to fill V# word2 (num_records). Astro Bot's
                // fullscreen-table VS uses it immediately before s_getpc_b64; treating every SOPK
                // as an unknown write prevented the otherwise-proven embedded table from folding.
                // The immediate is sign-extended by the ISA, matching the scalar register value.
                if (sopk_writes_scalar_data(in.opcode)) kill(in.dst.value);
                if (in.opcode == 0x00) {
                    kconst[in.dst.value] = static_cast<uint32_t>(static_cast<int32_t>(in.simm16));
                    fact_pc[in.dst.value] = in.pc;
                }
                break;
            case Rdna2Format::SMEM: {
                uint32_t n = 1;
                switch (in.opcode) { case 0x1: case 0x9: n=2; break; case 0x2: case 0xA: n=4; break;
                    case 0x3: case 0xB: n=8; break; case 0x4: case 0xC: n=16; break; default: break; }
                // Scalar-buffer form of the same compiler idiom: SBASE is the getpc-built V#, while
                // SOFFSET selects a row from the embedded table. Inspect it before killing SDATA facts.
                const int sb = in.src[0].value;
                if (in.opcode >= 0x8 && in.opcode <= 0xC && pcoff.count(sb) &&
                    pchi.count(sb + 1) && kconst.count(sb + 2) && kconst.count(sb + 3)) {
                    const uint32_t nrec = kconst[sb + 2];
                    const uint64_t off = pcoff[sb];
                    uint32_t newest = 0;
                    for (int r : {sb, sb + 1, sb + 2, sb + 3}) {
                        auto fact = fact_pc.find(r);
                        if (fact != fact_pc.end() && fact->second > newest) newest = fact->second;
                    }
                    bool entered = false;
                    for (uint32_t target : br_targets)
                        if (target > newest && target <= in.pc) { entered = true; break; }
                    const bool supported_soffset =
                        (in.src[1].kind == OperandKind::Special && in.src[1].value == 125) ||
                        in.src[1].kind == OperandKind::SGPR ||
                        (in.src[1].kind == OperandKind::Special && in.src[1].value >= 106 &&
                         in.src[1].value <= 123) ||
                        (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0);
                    if (!entered && supported_soffset && static_cast<int32_t>(in.literal) >= 0 &&
                        !(off & 3u) && !(nrec & 3u) && nrec && nrec <= 1024 &&
                        off / 4 + nrec / 4 <= dwords) {
                        if (required_dwords)
                            *required_dwords = std::max(*required_dwords,
                                static_cast<size_t>(off / 4 + nrec / 4));
                        out.smem[in.pc] = std::vector<uint32_t>(
                            code + off / 4, code + off / 4 + nrec / 4);
                    }
                }
                for (uint32_t k = 0; k < n; k++) kill(in.dst.value + (int)k);
                break;
            }
            case Rdna2Format::VOP1:
                if (in.opcode == 0x02) kill(in.dst.value);   // v_readfirstlane -> SGPR
                break;
            case Rdna2Format::VOPC:
                if (in.dst.kind == OperandKind::SGPR && in.dst.value <= 105)
                    { kill(in.dst.value); kill(in.dst.value + 1); }   // e64 compare -> SGPR pair
                break;
            case Rdna2Format::VOP3:
                if (in.sdst.kind == OperandKind::SGPR) { kill(in.sdst.value); kill(in.sdst.value + 1); }
                break;
            case Rdna2Format::MUBUF: {
                if (in.opcode < 0xCu || in.opcode > 0xFu) break;   // raw loads only
                const int sb = in.src[1].value;                    // SRSRC base SGPR of the V# quad
                if (!pcoff.count(sb) || !pchi.count(sb + 1) || !kconst.count(sb + 2)) break;
                const uint32_t nrec = kconst[sb + 2];              // V# word2 = num_records (bytes here)
                const bool offen = (in.literal >> 12) & 1u, idxen = (in.literal >> 13) & 1u;
                const bool soff0 = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                                   (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                const uint64_t off = pcoff[sb];
                // Entry soundness: no branch may enter between the newest contributing fact and here.
                uint32_t newest = 0;
                for (int r : {sb, sb + 1, sb + 2}) { auto it = fact_pc.find(r); if (it != fact_pc.end() && it->second > newest) newest = it->second; }
                bool entered = false;
                for (uint32_t t : br_targets) if (t > newest && t <= in.pc) { entered = true; break; }
                if (entered) break;
                if (idxen || !soff0 || (off & 3u) || (nrec & 3u) || nrec == 0 || nrec > 1024 ||
                    off / 4 + nrec / 4 > dwords) break;            // unprovable / out of window -> reject
                if (required_dwords)
                    *required_dwords = std::max(*required_dwords,
                                                static_cast<size_t>(off / 4 + nrec / 4));
                out.mubuf[in.pc] = std::vector<uint32_t>(code + off / 4, code + off / 4 + nrec / 4);
                (void)offen;                                       // offen just adds the runtime index
                break;
            }
            default: break;   // formats that don't write SGPRs
        }
    }
    return out;
}

inline bool reg_operand(const Operand& operand, int reg) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value == reg;
}

inline uint32_t scalar_branch_target(const Rdna2Inst& in) {
    return in.pc + in.len_dwords + static_cast<uint32_t>(in.simm16);
}

// Arbitrary CFG fallback. Some UE4 volume-lighting kernels and Astro Bot material shaders contain
// nested EXEC loops plus scalar/VCC branches. They are valid reducible machine CFGs, but deliberately
// exceed the narrow pattern structurizer below. Lower them as a structured dispatcher loop: one
// switch case per basic block, with the emulated register file persisted in Function variables
// between iterations. Graphics stages can branch directly on this invocation's SCC/VCC/EXEC bit;
// compute needs the guest-wave reduction described below.
// VCCZ/EXECZ reduce over the dispatch's 32/64-lane hardware wave, not the host Vulkan subgroup.
// Native subgroup widths are implementation-defined (llvmpipe is commonly 8), so every invocation
// remains in the dispatcher until the whole workgroup is done and exchanges its vote at the common
// switch merge. Dedicated Workgroup scratch keeps multiple hardware waves independent. The fallback
// is gated by emit_body to complex compute CFGs without guest barriers. Ordinary LDS reads, writes,
// and atomics remain valid while waves visit different cases. V_MBCNT is split into a dedicated
// common phase so every workgroup invocation reaches its synthesized barriers in uniform control flow.
// PROSPER_CFG_TRIP_BOUND=N — diagnostic only. Bound the CFG DISPATCHER's loop at N iterations, on
// whichever back-edge it emits (direct or portable). PROSPER_CFG_TRIP_BOUND_PROGRAM selects one
// program; PROSPER_CFG_TRIP_BOUND_PHASE selects one phase and is REQUIRED — see
// emitted_loop_trip_bound for why a record spanning two phases has no true reading. A program that
// accesses GDS itself is refused, because the witness lives in guest-addressable storage.
//
// It does NOT cover the two structured loop emitters. An earlier revision of this comment claimed it
// did; it never called this helper from either, so the claim was false and would have made a null
// result from a structured-loop program read as evidence.
//
// A loop that never terminates hangs the GPU into a driver reset, which costs the whole process its
// compute backend and every later indirect draw — and from outside it is indistinguishable from a
// slow shader or from a defect anywhere else in the submit. Bounding the dispatcher's loop turns
// "is this a non-terminating loop at all?" into one run for the programs it covers.
//
// (This paragraph used to end "and covering all three emitters means the answer is not confined to
// one lowering path" — the very claim the paragraph above corrects. The correction landed two lines
// up and the boast survived underneath it, which is how a retracted statement keeps being read as
// current. `tests/shared/diagnostics/test_cfg_trip_bound.cpp` now pins both halves: a structured loop is unchanged when
// armed, a dispatcher loop is not.)
//
// Unset, nothing is emitted and modules are byte-identical. It is NOT a fix: truncating a guest
// program's control flow produces wrong results by construction, which is why it is opt-in and says
// so when it arms.
// See kComputeTripWitnessDword. Deliberately ignores the phase selector: if ANY phase of this program
// will be bounded, the witness destination has to be bound for the whole dispatch.
// The complete selector state, read fresh rather than cached in function-local statics.
//
// Two reasons it is a struct and not three scattered getenv sites. First, the shader cache is keyed
// on the program's CODE BYTES and never on its address, so a target and a non-target with the same
// body collide on one entry: whichever compiled first would serve the other, handing the bounded
// module to an excluded program or the unbounded module to the target. Either direction silently
// destroys the isolation that makes a bounded run evidence about ONE program. The cache key therefore
// mixes this whole struct in (see gpu_executor.cpp), and a struct is what makes that complete by
// construction — a future selector added here cannot be forgotten there.
//
// Second, function-local statics parse the environment exactly once per process, which makes the
// selectors untestable in-process: no test could arm a bound, assert, then disarm and assert again.
// Re-reading costs three getenv calls per emitted loop during RECOMPILATION only (never per draw or
// per dispatch — cf. #2214, which removed per-resource-per-draw getenv from the live renderer), and
// recompiles are cache-warm after the first. CONFIDENCE: HIGH.
inline ComputeTripBoundSettings compute_trip_bound_settings() {
    ComputeTripBoundSettings settings;
    const char* spec = getenv("PROSPER_CFG_TRIP_BOUND");
    if (!spec || !*spec) return settings;
    char* end = nullptr;
    const unsigned long long parsed = strtoull(spec, &end, 0);
    if (!end || *end || !parsed || parsed > 0xffffffffull) return settings;
    settings.bound = static_cast<uint32_t>(parsed);

    // PROSPER_CFG_TRIP_BOUND_PROGRAM=0xADDR — restrict the bound to ONE program, leaving every other
    // recompiled module byte-identical. Without it, a bound low enough to be interesting truncates
    // MANY shaders, and any later change of behaviour has a second explanation: an earlier truncated
    // shader fed different data downstream. Targeting one program removes that alternative, which is
    // what turns "the run got further" into evidence about the program under test.
    if (const char* only = getenv("PROSPER_CFG_TRIP_BOUND_PROGRAM"); only && *only) {
        char* only_end = nullptr;
        const unsigned long long wanted = strtoull(only, &only_end, 0);
        if (only_end && !*only_end) settings.only_program = wanted;
    }
    // PROSPER_CFG_TRIP_BOUND_PHASE=K — bound only the K-th dispatcher of the selected program.
    //
    // A barrier-phased compute program emits one dispatcher per barrier-free phase, each covering a
    // different guest pc range. Bounding them together answers "does some loop in this program run
    // away" but not which, and they are not equivalent: an acyclic phase that needs a bound is a CFG
    // state-transition defect, while the phase containing the guest's own loop might merely be slow.
    if (const char* phase = getenv("PROSPER_CFG_TRIP_BOUND_PHASE"); phase && *phase) {
        char* phase_end = nullptr;
        const unsigned long long parsed_phase = strtoull(phase, &phase_end, 0);
        if (phase_end && !*phase_end && parsed_phase <= 0xfffffffeull)
            settings.only_phase = static_cast<uint32_t>(parsed_phase);
    }
    return settings;
}

struct BarrierPhasedCompute {
    size_t guard_index = 0;
    size_t end_index = 0;
    std::vector<size_t> barriers;
    bool guarded = false;
    bool found = false;
};

// Prove the SCC consumed by a terminal compute guard is identical for every wave in the workgroup,
// while allowing unrelated lane-local setup to precede it. The legacy barrier-phase proof required
// every instruction before the guard to be scalar; generated kernels commonly interleave vector
// address setup with a scalar descriptor load that independently produces the guard condition.
//
// This is deliberately a narrow backwards slice. The last SCC writer must be an SOPC, and every
// scalar word feeding that compare must trace through launch SGPRs, scalar ALU, or scalar memory.
// V_READFIRSTLANE/V_READLANE, vector mask writes, EXEC, and unresolved special registers terminate
// the proof. Existing all-scalar prefixes retain their established admission path below.
inline bool terminal_guard_scc_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins,
                                             size_t guard_index) {
    // This is a textual backwards slice, not a reaching-definition analysis. Keep the prefix
    // branch-free so every scalar definition it finds dominates the terminal guard. In particular,
    // a VCC/EXEC branch may vary between guest waves even when the literal or SMEM value written in
    // one arm is itself uniform.
    for (size_t i = 0; i < guard_index; ++i) {
        const Rdna2Inst& prefix = ins[i];
        if ((prefix.fmt == Rdna2Format::SOPP && !sopp_is_noop(prefix)) ||
            (prefix.fmt == Rdna2Format::SOP1 && prefix.opcode >= 0x20u &&
             prefix.opcode <= 0x22u))
            return false;
    }

    std::function<bool(int, uint32_t, uint32_t)> uniform_reg_at;
    auto operand_uniform_at = [&](const Operand& operand, uint32_t words, uint32_t use_pc,
                                  auto&& self) -> bool {
        switch (operand.kind) {
            case OperandKind::InlineInt:
            case OperandKind::InlineFloat:
            case OperandKind::Literal:
                return true;
            case OperandKind::Special:
                if (operand.value == 125) return true; // SGPR_NULL
                if (operand.value == 126 || operand.value == 127 || operand.value == 253)
                    return false; // EXEC and SCC are not scalar launch data.
                [[fallthrough]];
            case OperandKind::SGPR:
                for (uint32_t word = 0; word < words; ++word)
                    if (!self(operand.value + static_cast<int>(word), use_pc, 0)) return false;
                return true;
            default:
                return false;
        }
    };

    uniform_reg_at = [&](int reg, uint32_t use_pc, uint32_t depth) -> bool {
        if (depth > 64) return false;
        for (size_t i = ins.size(); i-- > 0;) {
            const Rdna2Inst& writer = ins[i];
            if (writer.pc >= use_pc) continue;
            bool writes = false;
            for_each_scalar_write(writer, [&](int base, uint32_t width) {
                writes |= reg >= base && reg < base + static_cast<int>(width);
            });
            // Implicit VOPC writes target architectural VCC even when the decoder exposes it as a
            // Special operand rather than an ordinary SGPR destination.
            if ((reg == 106 || reg == 107) && writer.fmt == Rdna2Format::VOPC &&
                !vopc_is_cmpx(writer.opcode))
                writes = true;
            if (!writes) continue;

            auto uniform_operand = [&](const Operand& operand, uint32_t words) {
                switch (operand.kind) {
                    case OperandKind::InlineInt:
                    case OperandKind::InlineFloat:
                    case OperandKind::Literal:
                        return true;
                    case OperandKind::Special:
                        if (operand.value == 125) return true;
                        if (operand.value == 126 || operand.value == 127 || operand.value == 253)
                            return false;
                        [[fallthrough]];
                    case OperandKind::SGPR:
                        for (uint32_t word = 0; word < words; ++word)
                            if (!uniform_reg_at(operand.value + static_cast<int>(word),
                                                writer.pc, depth + 1))
                                return false;
                        return true;
                    default:
                        return false;
                }
            };

            switch (writer.fmt) {
                case Rdna2Format::SOP1: {
                    // GETPC is identical for every wave. Mask/saveexec operations consume EXEC and
                    // are rejected by rdna2_instruction_may_change_exec; ordinary scalar operations trace
                    // their source at the destination's architectural width.
                    if (writer.opcode == 0x1f) return true;
                    if (rdna2_instruction_may_change_exec(writer) || writer.n_src != 1) return false;
                    return uniform_operand(writer.src[0], scalar_write_width(writer) == 2 ? 2u : 1u);
                }
                case Rdna2Format::SOP2: {
                    // Carry/cselect forms consume SCC in addition to their decoded operands. They are
                    // outside this narrow slice until SCC itself is represented as a recursive value.
                    if (writer.opcode == 0x04 || writer.opcode == 0x05 ||
                        writer.opcode == 0x0a || writer.opcode == 0x0b)
                        return false;
                    const uint32_t words = scalar_write_width(writer) == 2 ? 2u : 1u;
                    return writer.n_src == 2 && uniform_operand(writer.src[0], words) &&
                           uniform_operand(writer.src[1], words);
                }
                case Rdna2Format::SMEM: {
                    // Scalar-buffer loads use a four-dword descriptor; scalar-memory loads use a
                    // base pair. The scalar offset is one dword in both forms.
                    if (!scalar_write_width(writer) || writer.n_src < 1) return false;
                    const uint32_t base_words = writer.opcode >= 0x8 ? 4u : 2u;
                    if (!uniform_operand(writer.src[0], base_words)) return false;
                    return writer.n_src < 2 || uniform_operand(writer.src[1], 1);
                }
                case Rdna2Format::SOPK:
                    // S_MOVK is a literal definition. Other SOPK data writers are read/modify/write
                    // or hardware-register reads and stay outside this proof.
                    return writer.opcode == 0x00;
                default:
                    return false;
            }
        }
        // Ordinary compute launch SGPRs (user data and enabled system SGPRs) are uniform within one
        // workgroup. Special registers are numbered above this range and were handled explicitly.
        return reg >= 0 && reg <= 105;
    };

    for (size_t i = guard_index; i-- > 0;) {
        const Rdna2Inst& candidate = ins[i];
        if (sopp_is_noop(candidate)) continue;
        if (candidate.fmt == Rdna2Format::SOPC) {
            const uint32_t words = candidate.opcode == 0x12 || candidate.opcode == 0x13 ? 2u : 1u;
            for (uint32_t source = 0; source < candidate.n_src; ++source)
                if (!operand_uniform_at(candidate.src[source], words, candidate.pc,
                                        uniform_reg_at))
                    return false;
            return candidate.n_src != 0;
        }
        // Vector/memory operations cannot write SCC. Any scalar ALU or SOPK instruction might, so
        // stop rather than accidentally consuming an older compare through an unmodeled SCC writer.
        if (candidate.fmt == Rdna2Format::SOP1 || candidate.fmt == Rdna2Format::SOP2 ||
            candidate.fmt == Rdna2Format::SOPK)
            return false;
    }
    return false;
}

// Recognize barrier-separated compute phases. A generated kernel may wrap the phases in one proven
// workgroup-uniform terminal SCC guard, but the guard is not essential: an unconditional top-level
// barrier is itself a valid phase boundary. In either form, every explicit scalar edge must stay in
// one phase and indirect PC changes fail closed. Keeping this proof shared is important: emit_body
// uses it to split the shader, while the native subgroup policy uses the same result to make nested
// wave votes legal inside an individual phase.
inline BarrierPhasedCompute analyze_barrier_phased_compute(const std::vector<Rdna2Inst>& ins) {
    BarrierPhasedCompute result;
    result.end_index = ins.size();
    uint32_t end_pc = UINT32_MAX;
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!ins[i].is_end) continue;
        result.end_index = i;
        end_pc = ins[i].pc;
        break;
    }

    result.guard_index = ins.size();
    size_t first_barrier = ins.size();
    size_t branch_count = 0;
    for (size_t i = 0; i < result.end_index; ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a &&
            first_barrier == ins.size())
            first_barrier = i;
        if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03) continue;
        ++branch_count;
        if (result.guard_index == ins.size() &&
            (in.opcode == 0x04 || in.opcode == 0x05) && branch_target(in) == end_pc)
            result.guard_index = i;
    }

    bool guarded = result.guard_index < first_barrier && first_barrier < result.end_index;
    bool scalar_prefix = guarded;
    for (size_t i = 0; scalar_prefix && i < result.guard_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // Scalar ALU/loads retain workgroup-uniform values when entered from compute user data and
        // TGIDs. EXEC is lane state, so any explicit access to either half invalidates this proof.
        // VCC physical words may be scalar scratch provided no vector instruction enters the prefix.
        const bool scalar = in.fmt == Rdna2Format::SOP1 ||
            in.fmt == Rdna2Format::SOP2 || in.fmt == Rdna2Format::SOPK ||
            in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::SMEM ||
            (in.fmt == Rdna2Format::SOPP && sopp_is_noop(in));
        if (!scalar) {
            scalar_prefix = false;
            break;
        }
        auto is_exec = [](const Operand& operand) {
            return (operand.kind == OperandKind::SGPR ||
                    operand.kind == OperandKind::Special) &&
                   (operand.value == 126 || operand.value == 127);
        };
        if (is_exec(in.dst)) scalar_prefix = false;
        for (uint32_t source = 0; source < in.n_src; ++source)
            if (is_exec(in.src[source])) scalar_prefix = false;
    }
    if (guarded && !scalar_prefix)
        guarded = terminal_guard_scc_is_workgroup_uniform(ins, result.guard_index);

    // If a candidate terminal guard was not proved uniform, include it in the ordinary edge proof.
    // Since it jumps across every barrier to S_ENDPGM, that proof rejects the conditional barrier.
    // A proved guard is deliberately excluded: every invocation in the workgroup takes it together.
    result.guarded = guarded;
    const size_t body_begin = guarded ? result.guard_index + 1 : 0;
    bool valid = result.end_index < ins.size() && branch_count > 2;

    for (size_t i = body_begin; valid && i < result.end_index; ++i)
        if (ins[i].fmt == Rdna2Format::SOPP && ins[i].opcode == 0x0a)
            result.barriers.push_back(i);
    valid = valid && !result.barriers.empty();

    for (size_t i = body_begin; valid && i < result.end_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // A trap deactivates its guest wave in the dispatcher. Emitting a host workgroup barrier
        // afterward would then require an invocation that has left the phase to participate.
        // Final-phase traps are fine because no outer barrier follows them.
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12 &&
            i < result.barriers.back())
            valid = false;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u)
            valid = false;
    }
    // A phase boundary is valid only when no guest branch jumps across it in either direction.
    for (size_t barrier_index : result.barriers) {
        const uint32_t barrier_pc = ins[barrier_index].pc;
        for (size_t i = body_begin; valid && i < result.end_index; ++i) {
            const Rdna2Inst& in = ins[i];
            if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
                in.opcode == 0x03 || in.opcode == 0x0a) continue;
            const uint32_t target = branch_target(in);
            if ((in.pc < barrier_pc && target >= barrier_pc) ||
                (in.pc > barrier_pc && target <= barrier_pc))
                valid = false;
        }
    }
    result.found = valid;
    return result;
}

}  // namespace prosper::gpu

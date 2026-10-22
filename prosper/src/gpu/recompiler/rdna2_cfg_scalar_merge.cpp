// Existing scalar CFG worklist and merge-deadness result.
#include "gpu/recompiler/rdna2_cfg_scalar_merge.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_cfg_scalar_merge_routes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <optional>

namespace prosper::gpu {
bool sgpr_dead_at_merge(const std::vector<Rdna2Inst>& ins, uint32_t target, int R,
                        ScalarMergeProof proof, ScalarMergeBlocker* blocker,
                        // Explicit compare HIGH-word kills require the caller's pair-width
                        // proof. A Wave32 compare always replaces LOW but can leave HIGH live.
                        bool vopc_sgpr_pair_kills,
                        // Match physical mask writes to the caller's actual emitter mode.
                        ScalarMergeMaskWriteWidths mask_widths) {
    const auto block = [&](const Rdna2Inst& at, const char* kind) {
        if (blocker && blocker->pc == UINT32_MAX) {
            blocker->pc = at.pc;
            blocker->kind = kind;
        }
        return false;
    };
    // MaskDomainOnly is a VCC-half question by construction: an ordinary SGPR has no mask domain,
    // so the relaxation would silently become "reads are fine", which is not a proof of anything.
    if (proof == ScalarMergeProof::MaskDomainOnly && R != 106 && R != 107) return false;
    // M0 has readers that never name it: the relative moves, v_movrels_b32, the LDS and
    // append/consume forms, s_sendmsg. The operand scan below sees none of them, and a walk
    // that reached s_endpgm used to answer "dead" for it.
    if (R == 124) return false;
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
        if (in.is_end) continue;   // this path ends without a read
        // NOLINTNEXTLINE(bugprone-optional-value-conversion): engagement ends this worklist path.
        if (auto result =
                detail::CfgScalarMergeRoutes{in, R, data_read_ok, std::cref(block), pc_to_index,
                                             pending, index, ins, mask_widths, vopc_sgpr_pair_kills}
                    .step()) {
            if (!*result) return false;
            continue;
        }
        if (index + 1 < ins.size()) pending.push_back(index + 1);
    }
    return true;   // every reachable path hit a redefinition/end without a read
}

std::unordered_set<uint32_t> proven_cselect_b64_low_only_pcs(const std::vector<Rdna2Inst>& ins) {
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
        if (in.fmt == Rdna2Format::SOP2 && in.opcode == 0x0bu && in.dst.kind == OperandKind::SGPR &&
            in.dst.value == 106 && in.src[0].kind == OperandKind::SGPR &&
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
bool is_wave64_vcc_lo_scalar_b32_candidate(const Rdna2Inst& in) {
    const auto scalar_constant = [](const Operand& operand) {
        return operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat ||
               operand.kind == OperandKind::Literal;
    };
    const bool scalar_cselect_hi = in.fmt == Rdna2Format::SOP2 &&
                                   in.opcode == kSop2OpcodeCselectB32 &&
                                   in.dst.kind == OperandKind::SGPR && in.dst.value == 107 &&
                                   scalar_constant(in.src[0]) && scalar_constant(in.src[1]);
    return is_wave64_vcc_lo_scalar_cselect(in) || scalar_cselect_hi ||
           (in.fmt == Rdna2Format::SOP2 && sop2_is_b32_logical(in.opcode) &&
            in.dst.kind == OperandKind::SGPR && (in.dst.value == 106 || in.dst.value == 107));
}

std::unordered_set<uint32_t> proven_wave64_vcc_b32_low_only_pcs(const std::vector<Rdna2Inst>& ins,
                                                                bool include_logical,
                                                                bool include_high_cselect) {
    std::unordered_set<uint32_t> result;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) break;
        const bool candidate =
            (include_high_cselect && is_wave64_vcc_lo_cselect_reading_vcc_hi(in)) ||
            (include_logical ? is_wave64_vcc_lo_scalar_b32_candidate(in)
                             : is_wave64_vcc_lo_scalar_cselect(in));
        // Scalar scratch may use either physical VCC word and combine the two as ordinary dwords.
        // The sibling may be read as scalar data, but it must be dead before any pair/mask-domain
        // observation. MaskDomainOnly is exactly that whole-CFG proof.
        const int sibling = in.dst.value == 107 ? 106 : 107;
        if (candidate && sgpr_dead_at_merge(ins, in.pc + in.len_dwords, sibling,
                                            ScalarMergeProof::MaskDomainOnly))
            result.insert(in.pc);
    }
    return result;
}
}   // namespace prosper::gpu

#include "gpu/recompiler/rdna2_structured_saved_masks.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
StructuredSavedMaskEdge structured_saved_mask_edge(const SpirvCompute& b, const RegState& rs) {
    StructuredSavedMaskEdge edge{rs.sreg_bool, rs.sreg_bool_b32, {}};
    if (!ordinary_portable_wave64_masks(b)) return edge;
    for (const auto& [word, value] : rs.sreg)
        if (value && !rs.unavailable_saved_mask_words.contains(word) &&
            !rs.sreg_loop_blanket.contains(word) && !sreg_word_may_be_fabricated(rs, word))
            edge.current_data_words.insert(word);
    return edge;
}

bool join_structured_saved_masks(SpirvCompute& b, RegState& current,
                                 const StructuredSavedMaskEdge& other,
                                 const StructuredSavedMaskEdge& here, uint32_t other_block,
                                 uint32_t current_block, const std::vector<Rdna2Inst>& ins,
                                 uint32_t branch_pc, uint32_t merge_pc) {
    const bool carry_pair =
        !(b.allow_b32_masks && (b.is_fragment || (b.is_compute && b.wave_size == 32)));
    const ScalarMergeMaskWriteWidths widths{carry_pair, b.wave_size != 32, !b.allow_b32_masks};
    const auto dead = [&](int word) {
        return sgpr_dead_at_merge(ins, merge_pc, word, ScalarMergeProof::AnyRead, nullptr,
                                  carry_pair, widths);
    };
    std::set<int> keys;
    for (const auto& entry : other.predicates) keys.insert(entry.first);
    for (const auto& entry : here.predicates) keys.insert(entry.first);
    for (int key : keys) {
        const auto left = other.predicates.find(key);
        const auto right = here.predicates.find(key);
        const bool left_defined = left != other.predicates.end() && left->second;
        const bool right_defined = right != here.predicates.end() && right->second;
        const bool left_b32 = left_defined && other.b32.contains(key);
        const bool right_b32 = right_defined && here.b32.contains(key);
        if (!left_defined || !right_defined || left_b32 != right_b32) {
            // CONFIDENCE: HIGH -- a genuine complete DATA pair on BOTH predecessors already
            // has its ordinary scalar PHIs. Retire the competing Bool and let its later numeric
            // consumer read those actual words; do not project a Bool, seed a sibling or ballot.
            // This deliberately covers only ordinary portable Wave64, not B32/mixed-width views.
            const auto both_words = [&](int word) {
                return other.current_data_words.contains(word) &&
                       here.current_data_words.contains(word);
            };
            const bool numeric_pair =
                !left_b32 && !right_b32 && both_words(key) && both_words(key + 1);
            // CONFIDENCE: HIGH -- literal false loses genuine incoming bits (#4957).
            // Without both complete current DATA pairs, a missing Bool domain may hold an
            // incomplete pair or an undefined value. Neither a merge-filled scalar nor an old
            // entry spelling proves a predecessor value; no Bool projection is invented here.
            // A B32 predicate owns only LOW. Its neighboring physical DATA word can remain live.
            // Conversely, any surviving B64 edge still owns HIGH even if the other edge is B32.
            const bool owns_high = (left_defined && !left_b32) || (right_defined && !right_b32);
            if (!numeric_pair && (!dead(key) || (owns_high && !dead(key + 1)))) {
                b.stage_reject_pc = branch_pc;
                b.stage_reject_reason = "structured-saved-mask-domain-missing-edge";
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "reason=structured-saved-mask-domain-missing-edge "
                                         "pc=%u merge=%u s%d",
                                         branch_pc, merge_pc, key);
                return false;
            }
            // Genuine current DATA or a complete later replacement needs no old predicate phi.
            current.sreg_bool.erase(key);
            current.sreg_bool_narrowed.erase(key);
            current.sreg_bool_b32.erase(key);
            if (key == 106) current.vcc = 0;
            continue;
        }
        if (left->second != right->second) {
            current.sreg_bool[key] =
                b.emit_phi_2way(b.t_bool, left->second, other_block, right->second, current_block);
            current.sreg_bool_narrowed[key] = true;
        }
    }
    return true;
}
}   // namespace prosper::gpu

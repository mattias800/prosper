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
bool join_structured_saved_masks(SpirvCompute& b, RegState& current,
                                 const std::unordered_map<int, uint32_t>& other,
                                 const std::unordered_set<int>& other_b32, uint32_t other_block,
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
    for (const auto& entry : other) keys.insert(entry.first);
    for (const auto& entry : current.sreg_bool) keys.insert(entry.first);
    for (int key : keys) {
        const auto left = other.find(key);
        const auto right = current.sreg_bool.find(key);
        const bool left_defined = left != other.end() && left->second;
        const bool right_defined = right != current.sreg_bool.end() && right->second;
        const bool left_b32 = left_defined && other_b32.contains(key);
        const bool right_b32 = right_defined && current.sreg_bool_b32.contains(key);
        if (!left_defined || !right_defined || left_b32 != right_b32) {
            // CONFIDENCE: HIGH -- literal false loses genuine incoming bits (#4957).
            // A missing Bool domain may instead hold DATA, an incomplete word pair or an
            // undefined value. This join has no predecessor projection/validity protocol;
            // neither a merge-filled scalar nor its old entry spelling supplies one.
            // A B32 predicate owns only LOW. Its neighboring physical DATA word can remain live.
            // Conversely, any surviving B64 edge still owns HIGH even if the other edge is B32.
            const bool owns_high = (left_defined && !left_b32) || (right_defined && !right_b32);
            if (!dead(key) || (owns_high && !dead(key + 1))) {
                b.stage_reject_pc = branch_pc;
                b.stage_reject_reason = "structured-saved-mask-domain-missing-edge";
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "reason=structured-saved-mask-domain-missing-edge "
                                         "pc=%u merge=%u s%d",
                                         branch_pc, merge_pc, key);
                return false;
            }
            // A complete later replacement on every path needs no old predicate phi.
            current.sreg_bool.erase(key);
            current.sreg_bool_narrowed.erase(key);
            current.sreg_bool_b32.erase(key);
            if (key == 106) current.vcc = 0;
            continue;
        }
        if (left->second != right->second) {
            right->second =
                b.emit_phi_2way(b.t_bool, left->second, other_block, right->second, current_block);
            current.sreg_bool_narrowed[key] = true;
        }
    }
    return true;
}
}   // namespace prosper::gpu

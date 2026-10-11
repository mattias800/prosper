#include "gpu/recompiler/rdna2_structured_saved_masks.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <set>
#include <unordered_map>
#include <vector>

namespace prosper::gpu {
bool join_structured_saved_masks(SpirvCompute& b, RegState& current,
                                 const std::unordered_map<int, uint32_t>& other,
                                 uint32_t other_block, uint32_t current_block,
                                 const std::vector<Rdna2Inst>& ins, uint32_t branch_pc,
                                 uint32_t merge_pc) {
    std::set<int> keys;
    for (const auto& entry : other) keys.insert(entry.first);
    for (const auto& entry : current.sreg_bool) keys.insert(entry.first);
    for (int key : keys) {
        const auto left = other.find(key);
        const auto right = current.sreg_bool.find(key);
        if (left == other.end() || right == current.sreg_bool.end() || !left->second ||
            !right->second) {
            // CONFIDENCE: HIGH -- literal false loses genuine incoming bits (#4957).
            // A missing Bool domain may instead hold DATA, an incomplete word pair or an
            // undefined value. This join has no predecessor projection/validity protocol;
            // neither a merge-filled scalar nor its old entry spelling supplies one.
            if (!sgpr_dead_at_merge(ins, merge_pc, key) ||
                !sgpr_dead_at_merge(ins, merge_pc, key + 1)) {
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

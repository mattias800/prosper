// See rdna2_spill_slot_domain.hpp.
#include "gpu/recompiler/rdna2_spill_slot_domain.hpp"

#include <cstdint>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace prosper::gpu {
namespace {

constexpr uint32_t kOpReadlane = 0x360;
constexpr uint32_t kOpWritelane = 0x361;

bool constant_lane(const Rdna2Inst& in) {
    return in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 &&
           in.src[1].value <= 63;
}

SpillSlotDomain join(SpillSlotDomain a, SpillSlotDomain b) {
    if (a == b) return a;
    if (a == SpillSlotDomain::Conflict || b == SpillSlotDomain::Conflict)
        return SpillSlotDomain::Conflict;
    if (is_mask_domain(a) && is_mask_domain(b)) return SpillSlotDomain::MaskAny;
    if (is_mask_domain(a) || is_mask_domain(b)) return SpillSlotDomain::Conflict;
    return SpillSlotDomain::Other;   // Data and Other
}

// What emit_alu's V_WRITELANE stores, decided the way it decides: a Bool for EXEC, for a VCC word
// with no scalar value, for a proven reloaded mask half, and for an SGPR holding a Bool; the
// source's 32-bit value otherwise.
SpillSlotDomain written_domain(const Rdna2Inst& in, const SpillSlotMustSets& sets,
                               const SpillSlotContext& context) {
    if (context.proven_half_writelane_pcs.contains(in.pc)) return SpillSlotDomain::MaskAny;
    const Operand& source = in.src[0];
    switch (source.kind) {
        case OperandKind::InlineInt:
        case OperandKind::InlineFloat:
        case OperandKind::Literal: return SpillSlotDomain::Data;
        case OperandKind::SGPR:
        case OperandKind::Special: break;
        default: return SpillSlotDomain::Other;
    }
    if (source.value == 126) return SpillSlotDomain::MaskLo;
    if (source.value == 127) return SpillSlotDomain::MaskHi;
    if (source.value == 106 || source.value == 107) {
        // A VCC word with a scalar value is stored as data, unless it may also hold a Bool.
        if (!sets.scalar_words.contains(source.value)) return SpillSlotDomain::MaskAny;
        return sets.masks.contains(source.value) ? SpillSlotDomain::Other : SpillSlotDomain::Data;
    }
    if (source.kind == OperandKind::SGPR && sets.masks.contains(source.value))
        // A mask the dispatcher cannot persist as a Bool reaches the write as a Bool only when it
        // was defined in the same case. Which one the write sees is not decidable here.
        return context.mask_keys.contains(source.value) ? SpillSlotDomain::MaskLo
                                                        : SpillSlotDomain::Conflict;
    if (spill_source_is_scalar_data(in, sets.masks, sets.ambiguous, sets.scalar_words,
                                    sets.readlane_words))
        return SpillSlotDomain::Data;
    return SpillSlotDomain::Other;
}

void forget_vgpr(SpillSlotDomains& state, int vgpr) {
    for (auto it = state.lower_bound({vgpr, -1}); it != state.end() && it->first.first == vgpr;)
        it = state.erase(it);
}

// A V_READLANE from a typed slot. Its generic transfer already made the destination a scalar word.
void retype_reload(const SpillSlotDomains& state, const Rdna2Inst& in,
                   const SpillSlotMustSets& sets, const SpillSlotContext& context) {
    // A proven mask-half reload publishes the exact ballot word as scalar data (emit_alu).
    if (!constant_lane(in) || context.proven_half_readlane_pcs.contains(in.pc)) return;
    const auto slot = state.find({in.src[0].value, in.src[1].value});
    if (slot == state.end()) return;
    const int dst = in.dst.value;
    const SpillSlotDomain domain = slot->second;
    if (domain == SpillSlotDomain::Data) {
        sets.readlane_words.erase(dst);   // a typed data reload: re-spilling it stores data
        return;
    }
    if (domain == SpillSlotDomain::Other) return;
    // emit_alu gives a mask reload only a Bool, keyed on the destination as the low word of a
    // pair. Its uint variable is a placeholder, so it is not a scalar word. A native data read
    // takes the Bool's low ballot word, which is the slot's dword only for a MaskLo slot; a portable
    // one is refused. A mask on only some paths is ambiguous: the first read of it is refused.
    sets.scalar_words.erase(dst);
    const bool exact_mask =
        domain == SpillSlotDomain::MaskLo || (is_mask_domain(domain) && !context.native_wave64);
    if (exact_mask && dst <= 105 && context.mask_keys.contains(dst)) {
        sets.masks.insert(dst);
        sets.ambiguous.erase(dst);
    } else {
        sets.ambiguous.insert(dst);
    }
}

}   // namespace

bool is_mask_domain(SpillSlotDomain domain) {
    return domain == SpillSlotDomain::MaskLo || domain == SpillSlotDomain::MaskHi ||
           domain == SpillSlotDomain::MaskAny;
}

void meet_spill_slot_domains(SpillSlotDomains& into, const SpillSlotDomains& incoming) {
    for (const auto& [slot, domain] : incoming) {
        const auto found = into.find(slot);
        if (found == into.end())
            into.emplace(slot, domain);
        else
            found->second = join(found->second, domain);
    }
}

bool spill_source_is_scalar_data(const Rdna2Inst& in, const std::set<int>& masks,
                                 const std::set<int>& ambiguous, const std::set<int>& scalar_words,
                                 const std::set<int>& readlane_words) {
    if (in.src[0].kind != OperandKind::SGPR) return false;
    const int source = in.src[0].value;
    const auto in_pair = [&](const std::set<int>& pairs) {
        return pairs.contains(source) || (source > 0 && pairs.contains(source - 1));
    };
    return scalar_words.contains(source) && !in_pair(masks) && !in_pair(ambiguous) &&
           !readlane_words.contains(source);
}

void advance_spill_slot_domains(SpillSlotDomains& state, const Rdna2Inst& in,
                                const SpillSlotMustSets& sets, const SpillSlotContext& context,
                                const std::vector<int>& vector_writes,
                                std::unordered_set<uint32_t>* mask_writelane_pcs) {
    const bool lane_op =
        in.fmt == Rdna2Format::VOP3 && (in.opcode == kOpWritelane || in.opcode == kOpReadlane);
    if (!lane_op) {
        for (int vgpr : vector_writes) forget_vgpr(state, vgpr);
        return;
    }
    if (in.opcode == kOpReadlane) {
        retype_reload(state, in, sets, context);
        return;
    }
    if (!constant_lane(in)) {
        forget_vgpr(state, in.dst.value);   // a dynamic lane may replace any slot
        return;
    }
    const SpillSlotDomain domain = written_domain(in, sets, context);
    state[{in.dst.value, in.src[1].value}] = domain;
    if (mask_writelane_pcs && is_mask_domain(domain)) mask_writelane_pcs->insert(in.pc);
}

}   // namespace prosper::gpu

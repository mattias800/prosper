#include "gpu/recompiler/rdna2_portable_mask_dispatch.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_portable_mask_slots.hpp"
#include "gpu/recompiler/rdna2_spill_slot_domain.hpp"
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <utility>

namespace prosper::gpu {
namespace {
const PortableMaskHalfTypes* before(const SpirvCompute& b, uint32_t pc) {
    if (!ordinary_portable_wave64_masks(b) || !b.portable_mask_half_program) return nullptr;
    const auto found = b.portable_mask_half_program->before.find(pc);
    return found == b.portable_mask_half_program->before.end() ? nullptr : &found->second;
}
bool static_lane(const Rdna2Inst& in) {
    return in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 && in.src[1].value < 64;
}
bool pair(const PortableMaskHalfTypes& types, int root) {
    const auto low = types.words.find(root), high = types.words.find(root + 1);
    return low != types.words.end() && high != types.words.end() && low->second == 0 &&
           high->second == 1;
}
bool complete_active(const SpirvCompute& b, uint32_t value) {
    if (!value) return true; // the caller's ordinary unguarded entry
    // btrue() emits a fresh ID each time. Authenticate its actual typed declaration rather
    // than comparing IDs or folding an opaque runtime/specification-dependent predicate.
    for (size_t at = 0; at < b.types.size();) {
        const uint32_t words = b.types[at] >> 16u;
        if (!words || words > b.types.size() - at) return false;
        if ((b.types[at] & 0xffffu) == Op_ConstantTrue && words == 3 &&
            b.types[at + 1] == b.t_bool && b.types[at + 2] == value)
            return true;
        at += words;
    }
    return false;
}
SpillSlotDomain domain(uint32_t role) {
    return role == 0 ? SpillSlotDomain::MaskLo : SpillSlotDomain::MaskHi;
}
} // namespace

PortableMaskHalfEntryScope::PortableMaskHalfEntryScope(SpirvCompute& b, uint32_t active)
    : b_(b), previous_(b.portable_mask_common_partial_entry) {
    b.portable_mask_common_partial_entry = previous_ || !complete_active(b, active);
}
PortableMaskHalfEntryScope::~PortableMaskHalfEntryScope() {
    b_.portable_mask_common_partial_entry = previous_;
}

void refine_portable_mask_dispatch(const SpirvCompute& b, const Rdna2Inst& in, bool exact_wave_scc,
                                   const SpilledVectorSourceMarks& source, SpilledMaskHalves& marks,
                                   SpillSlotDomains& domains) {
    const auto* types = before(b, in.pc);
    if (!types || source.memory) return;
    // These are the same closed symbolic publishers as PortableMaskHalfInstruction. Its actual
    // carrier/mark checks still run during emission; no uint word becomes defined here.
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 && static_lane(in) && in.dst.value >= 0 &&
        in.dst.value <= 105 && in.dst.kind == OperandKind::VGPR &&
        types->slots.contains({in.src[0].value, in.src[1].value}))
        marks.relayed_words.erase(in.dst.value);
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 &&
        in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value <= 105 &&
        in.src[0].kind == OperandKind::SGPR && types->words.contains(in.src[0].value))
        marks.relayed_words.erase(in.dst.value);
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361 && static_lane(in)) {
        const auto after = b.portable_mask_half_program->after.find(in.pc);
        const std::pair<int, int> slot{in.dst.value, in.src[1].value};
        if (after != b.portable_mask_half_program->after.end())
            if (const auto role = after->second.slots.find(slot);
                role != after->second.slots.end() && role->second <= 1) {
                marks.defined_slots.insert(slot);
                domains[slot] = domain(role->second);
            }
    }
    // Only a synchronized exact B64 comparison over current symbolic pairs can replace this
    // SCC mark. Missing/role-conflicting pairs, other consumers and native ownership stay intact.
    if (!exact_wave_scc || in.fmt != Rdna2Format::SOPC || (in.opcode != 0x12 && in.opcode != 0x13))
        return;
    bool symbolic = false;
    for (uint32_t k = 0; k < in.n_src; ++k) {
        const auto& operand = in.src[k];
        if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) continue;
        const int root = operand.value;
        if (!types->may_words.contains(root) && !types->may_words.contains(root + 1)) {
            // The typed operand cannot repair a different legacy operand. The caller's exact
            // classification and that operand's existing per-word ownership must both survive.
            for (int word : {root, root + 1})
                if (marks.relayed_words.contains(word) || marks.memory_words.contains(word)) return;
            continue;
        }
        symbolic = true;
        if (!pair(*types, root)) return;
    }
    if (symbolic) marks.relayed_words.erase(kRelayedScc);
}
}   // namespace prosper::gpu

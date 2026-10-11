// #4935: dispatcher services must consume the same per-word authority as emit_alu.
// CONFIDENCE: HIGH -- every query names the original consumed PC and current scalar carrier.
#include "gpu/recompiler/rdna2_portable_mask_slots.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <utility>

namespace prosper::gpu {
namespace {
const PortableMaskHalfTypes* types_at(const SpirvCompute& b, uint32_t pc) {
    if (!ordinary_portable_wave64_masks(b) || !b.portable_mask_half_program) return nullptr;
    const auto found = b.portable_mask_half_program->before.find(pc);
    return found == b.portable_mask_half_program->before.end() ? nullptr : &found->second;
}
bool half_overlap(const PortableMaskHalfTypes* types, int source, uint32_t width) {
    if (!types) return false;
    for (uint32_t word = 0; word < width; ++word)
        if (types->may_words.contains(source + static_cast<int>(word))) return true;
    return false;
}
bool projection_tainted(const PortableMaskHalfTypes* types, int source) {
    // A lone LOW carrier must not masquerade as the adjacent word's missing HIGH source.
    // An independent MUST complete root is its own current whole-mask generation.
    return half_overlap(types, source, 1) ||
           (half_overlap(types, source - 1, 1) && !types->numeric_words.contains(source) &&
            !types->complete.contains(source));
}
uint32_t half_value(const RegState& rs, const PortableMaskHalfTypes& types, int source) {
    const auto role = types.words.find(source);
    const auto value = rs.sreg_bool.find(source);
    if (role == types.words.end() || role->second > 1 || value == rs.sreg_bool.end() ||
        rs.sreg_merge_placeholder.contains(source) || rs.sreg_memory_pattern.contains(source))
        return 0;
    return value->second;
}
bool pair(const PortableMaskHalfTypes& types, int root) {
    const auto low = types.words.find(root), high = types.words.find(root + 1);
    return low != types.words.end() && high != types.words.end() && low->second == 0 &&
           high->second == 1;
}
} // namespace
bool portable_mask_common_pair_is_proven(const SpirvCompute& b, const Rdna2Inst& in, int root,
                                         bool legacy) {
    const auto* types = types_at(b, in.pc);
    return half_overlap(types, root, 2) ? pair(*types, root) : legacy;
}
const char* prepare_portable_mask_common_sources(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                                 bool incomplete_entry) {
    const auto* types = types_at(b, in.pc);
    if (!types) return nullptr;
    const bool b64 = (in.fmt == Rdna2Format::SOPC && (in.opcode == 0x12 || in.opcode == 0x13)) ||
                     (in.fmt == Rdna2Format::VOPC && in.opcode >= 0xe1 && in.opcode <= 0xe6) ||
                     (in.fmt == Rdna2Format::SOP1 &&
                      (in.opcode == kSop1OpcodeBcnt1I32B64 || in.opcode == kSop1OpcodeFf1I32B64));
    const bool ffbh = in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeFfbhU32;
    const bool mbcnt = in.fmt == Rdna2Format::VOP3 && (in.opcode == 0x365 || in.opcode == 0x366);
    if (!b64 && !ffbh && !mbcnt) return nullptr;
    for (uint32_t i = 0; i < in.n_src; ++i) {
        const auto& source = in.src[i];
        if (source.kind != OperandKind::SGPR && source.kind != OperandKind::Special) continue;
        if (!(b64 ? half_overlap(types, source.value, 2) : projection_tainted(types, source.value)))
            continue;
        // A per-invocation Bool cannot supply scalar bits belonging to unlaunched lanes.
        // Restrict only this new symbolic-half service; ordinary DATA and legacy whole-mask
        // routes retain their existing contracts.
        if (b.local_count < 64 || b.local_count % 64 != 0 || b.portable_mask_common_partial_entry ||
            incomplete_entry)
            return "portable-mask-half-incomplete-wave";
        if (b64) {
            const uint32_t value = portable_mask_half_pair_value(b, rs, in.pc, source.value);
            if (!value) return "portable-mask-half-pair-unavailable";
            // The LOW carrier's HIGH bits are unobservable as a LOW word. Normalizing them to
            // the actual HIGH predicate lets an existing common B64 service consume the pair,
            // without publishing DATA or changing either word's role/current selected bits.
            rs.sreg_bool[source.value] = value;
            for (int word : {source.value, source.value + 1}) {
                rs.sreg.erase(word);
                rs.sreg_input.erase(word);
                rs.sreg_srt.erase(word);
            }
        } else if (i != 0 || !half_value(rs, *types, source.value) ||
                   (mbcnt && types->words.at(source.value) != (in.opcode == 0x366 ? 1u : 0u)))
            // A HIGH dword used as MBCNT_LO needs a separate bit-position remapping service.
            // The existing phase scans canonical halves only; it cannot grant that DATA view.
            return "portable-mask-half-projection-unavailable";
    }
    rs.portable_mask_half_types = b.portable_mask_half_program->after.at(in.pc);
    return nullptr;
}
std::pair<uint32_t, uint32_t>
portable_mask_common_ffbh_source(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in) {
    const int source = in.src[0].value;
    const auto* types = types_at(b, in.pc);
    if (projection_tainted(types, source)) {
        const uint32_t value = half_value(rs, *types, source);
        return value ? std::pair{value, types->words.at(source)} : std::pair<uint32_t, uint32_t>{};
    }
    int root = source;
    auto value = rs.sreg_bool.find(root);
    if (value == rs.sreg_bool.end() || rs.sreg_bool_b32.contains(root)) {
        root = source - 1;
        value = rs.sreg_bool.find(root);
    }
    if (root < 0 || source - root < 0 || source - root > 1 || value == rs.sreg_bool.end() ||
        rs.sreg_bool_b32.contains(root))
        return {};
    return {value->second, static_cast<uint32_t>(source - root)};
}
uint32_t portable_mask_common_mbcnt_source(const SpirvCompute& b, const RegState& rs,
                                           const Rdna2Inst& in, int legacy_root) {
    const auto* types = types_at(b, in.pc);
    const int source = in.src[0].value;
    if (projection_tainted(types, source)) {
        const uint32_t value = half_value(rs, *types, source);
        return value && types->words.at(source) == (in.opcode == 0x366 ? 1u : 0u) ? value : 0;
    }
    const auto value = rs.sreg_bool.find(legacy_root);
    return value == rs.sreg_bool.end() ? 0 : value->second;
}
}   // namespace prosper::gpu

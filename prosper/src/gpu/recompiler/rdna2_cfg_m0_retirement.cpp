#include "gpu/recompiler/rdna2_cfg_m0_retirement.hpp"
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>
namespace prosper::gpu {
namespace {
bool names_word(const Operand& operand, int word) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value == word;
}
bool interval_has_m0(int first, uint32_t width) {
    return first >= 0 && first <= 124 && width > static_cast<uint32_t>(124 - first);
}
bool prefix_has_no_m0_use(const Rdna2Inst& in) {
    const bool scalar = in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                        in.fmt == Rdna2Format::SOPK || in.fmt == Rdna2Format::SOPC ||
                        in.fmt == Rdna2Format::SMEM;
    const bool valu = in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 ||
                      in.fmt == Rdna2Format::VOP3 || in.fmt == Rdna2Format::VOP3P ||
                      in.fmt == Rdna2Format::VOPC;
    const auto& effects = rdna2_instruction_effects(in);
    // Store/atomic SDATA is an implicit input outside this legacy source inventory.
    if (in.fmt == Rdna2Format::SMEM && !rdna2_smem_load_dwords(in.opcode)) return false;
    if ((!scalar && !valu) || (scalar && !effects.known()) ||
        (valu && !effects.describes_sources()) || effects.control != Rdna2ControlFlow::None ||
        (effects.implicit_reads & kRdna2ImplicitM0) ||
        effects.m0_relative != Rdna2M0Relative::None || names_word(in.dst, 124))
        return false;
    bool writes_m0 = false;
    for_each_scalar_write(
        in, [&](int first, uint32_t width) { writes_m0 |= interval_has_m0(first, width); });
    if (writes_m0 || interval_has_m0(in.dst.value, scalar_implicit_destination_read_width(in)))
        return false;
    for (uint32_t k = 0; k < in.n_src; ++k) {
        const auto& operand = in.src[k];
        if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) continue;
        const uint32_t width = effects.ssrc_dwords[k];
        if (!width) continue; // known row: the decoded field is not an operand
        if (width > 128 || interval_has_m0(operand.value, width)) return false;
    }
    return true;
}
} // namespace
bool deferred_waterfall_m0_retirement(const std::vector<Rdna2Inst>& ins, std::size_t first,
                                      int index) {
    if (first >= ins.size() || index < 0 || index > 105) return false;
    const auto end = first + std::min(ins.size() - first, std::size_t{64});
    uint32_t next = ins[first].pc;
    for (auto at = first; at < end; ++at) {
        const auto& in = ins[at];
        if (in.pc != next || !in.len_dwords || in.len_dwords > std::size(in.words) || in.is_end ||
            in.synthetic_terminator || in.len_dwords > UINT32_MAX - in.pc ||
            rdna2_escapes_decoded_effects(in))
            return false;
        uint32_t packet[5];
        std::copy(std::begin(in.words), std::end(in.words), std::begin(packet));
        // Trailing literals are stored separately from the encoding words.
        if (in.has_literal) {
            if (in.len_dwords < 2) return false;
            packet[in.len_dwords - 1] = in.literal;
        }
        auto decoded = rdna2_decode_one(packet, in.len_dwords);
        decoded.pc = in.pc;
        if (decoded != in) return false;
        next += in.len_dwords;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 &&
            names_word(in.dst, 124))
            return !names_word(in.src[0], 124) && !names_word(in.src[0], index);
        if (in.fmt == Rdna2Format::SOPP) {
            if (!sopp_is_noop(in)) return false;
        } else if (!prefix_has_no_m0_use(in)) {
            return false;
        }
    }
    return false;
}
} // namespace prosper::gpu

#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <cstdint>

namespace prosper::gpu {

// DS decoding retains all three encoded VGPR fields and its storage inventory uses a
// conservative four-word range. These full-result services consume only the following words;
// reserved DATA fields and adjacent registers are not dependencies of a read or swizzle.
inline uint32_t ordinary_vgpr_consumed_source_words(const Rdna2Inst& in, uint32_t source_index) {
    if (source_index >= in.n_src || in.src[source_index].kind != OperandKind::VGPR) return 0;
    if (in.fmt == Rdna2Format::DS) {
        switch (in.opcode) {
            case 0x35u:
            case 0x36u:
            case 0x37u:
            case 0x38u:
            case 0x76u:
            case 0x77u:
            case 0xfeu:
            case 0xffu: return source_index == 0u ? 1u : 0u;
            case 0x3du:
            case 0x3eu:
            case 0xb1u: return 0; // append/consume and ADDTID read use M0, not VGPR fields.
            case 0x20u:
            case 0x2du: // WRXCHG return consumes ADDR/DATA0; DATA1 is reserved.
            case kDsOpcodeBpermuteB32: return source_index <= 1u ? 1u : 0u;
            default: break;
        }
    }
    if ((in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) && source_index == 0u) {
        const bool offen = (in.literal & (1u << 12u)) != 0u;
        const bool idxen = (in.literal & (1u << 13u)) != 0u;
        return static_cast<uint32_t>(offen) + static_cast<uint32_t>(idxen);
    }
    if (in.fmt == Rdna2Format::FLAT && source_index == 0u) return in.flat_segment == 1u ? 1u : 2u;
    if (in.fmt == Rdna2Format::MIMG && source_index == 0u) {
        // This local lifetime proof covers actual ordinary 2D word addresses. Other dimensional
        // and packed forms need their own consumed-word inventory before expiring a spill.
        if (in.opcode == 0x0eu) return 1u;   // GET_RESINFO reads only the LOD.
        // Storage images read ordinary coordinate words even when A16 is encoded; sampled
        // textures unpack that form. A resource-independent effect cannot assume either span.
        if (in.mimg_dim != 1u || in.mimg_a16) return UINT32_MAX;
        switch (in.opcode) {
            case 0x00u: return 2u;
            case 0x01u: return 3u;
            case 0x20u:
            case 0xa0u:
            case 0x27u:
            case 0x40u:
            case 0x47u:
            case 0x60u: return 2u;
            case 0x24u:
            case 0x25u:
            case 0x28u:
            case 0x2fu:
            case 0x30u:
            case 0x37u:
            case 0x57u: return 3u;
            case 0x22u:
            case 0x68u: return 6u;
            default: return UINT32_MAX;
        }
    }
    return rdna2_vgpr_source_span(in, source_index);
}

// NSA MIMG addresses are explicit encoded VGPR words, not a consecutive span from VADDR.
inline int ordinary_vgpr_consumed_source_register(const Rdna2Inst& in, uint32_t source_index,
                                                  uint32_t word) {
    if (in.fmt == Rdna2Format::MIMG && source_index == 0u && in.len_dwords > 2u && word) {
        const uint32_t encoded_word = 2u + (word - 1u) / 4u;
        if (encoded_word >= in.len_dwords) return -1;
        return static_cast<int>((in.words[encoded_word] >> (8u * ((word - 1u) % 4u))) & 0xffu);
    }
    return in.src[source_index].value + static_cast<int>(word);
}

// Complete ordinary destination words in the supported lowering, independent of the old VDST.
// This is a bit-definition effect, not source initialization or EXEC authority. The caller must
// prove full EXEC and the emitter must still resolve every consumed operand. Unknown, relative,
// DPP, preserving SDWA, scalar-half and implicit-accumulator results supply no such effect.
inline uint32_t ordinary_vgpr_complete_results(const Rdna2Inst& in) {
    if (in.dst.kind != OperandKind::VGPR || in.dst.value < 0 || in.has_dpp ||
        (in.has_sdwa && (in.sdwa_dst_sel != 6u || in.sdwa_dst_unused == 2u)))
        return 0;
    switch (in.fmt) {
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
        case Rdna2Format::VOP3:
            // The effects table: a lane-wise result of the explicit operands alone, in every
            // VGPR VDST receives. Accumulators (MAC/FMAC/DOT*C read the old VDST), 16-bit results
            // outside the zero-filling SDWA PAD form, and cross-lane, selected-lane, relative and
            // interpolation results supply no such effect. A valid opcode with no emitter is
            // refused at emission, so naming it here grants nothing.
            return rdna2_valu_complete_result_dwords(in);
        case Rdna2Format::VOP3P:
            // Packed arithmetic/dot results write both halves or a full F32 word; MIXLO/MIXHI
            // preserve the opposite half and are deliberately absent.
            return in.opcode <= 0x12u || (in.opcode >= 0x14u && in.opcode <= 0x19u) ||
                           in.opcode == 0x20u
                       ? 1u
                       : 0u;
        case Rdna2Format::DS:
            // Full-word reads/returning atomics and swizzles. D16 preserving reads are absent.
            switch (in.opcode) {
                case 0x20u:
                case 0x2du:
                case 0x35u:
                case 0x36u:
                case 0x3du:
                case 0x3eu:
                case 0xb1u:
                case kDsOpcodeBpermuteB32: return 1;
                case 0x37u:
                case 0x38u:
                case 0x76u: return 2;
                case 0xfeu: return 3;
                case 0x77u:
                case 0xffu: return 4;
                default: return 0;
            }
        case Rdna2Format::MUBUF:
            return !in.mubuf_lds &&
                           (in.opcode <= 0x03u || (in.opcode >= 0x08u && in.opcode <= 0x0fu))
                       ? rdna2_vgpr_write_count(in)
                       : 0u;
        case Rdna2Format::MTBUF: return in.opcode <= 0x03u ? rdna2_vgpr_write_count(in) : 0u;
        case Rdna2Format::MIMG:
            return !in.mimg_d16 && in.mimg_dmask &&
                           ordinary_vgpr_consumed_source_words(in, 0u) <= 256u
                       ? rdna2_vgpr_write_count(in)
                       : 0u;
        case Rdna2Format::FLAT:
            return in.opcode >= 0x08u && in.opcode <= 0x0fu ? rdna2_vgpr_write_count(in) : 0u;
        default: return 0;
    }
}

}   // namespace prosper::gpu

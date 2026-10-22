// Original scalar CFG case groups; native path exits are retained.
#include <cstdint>
#include "gpu/recompiler/rdna2_cfg_scalar_merge_routes.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <optional>

namespace prosper::gpu::detail {
std::optional<bool> CfgScalarMergeRoutes::smem() {
    do {
        switch (in.fmt) {
            case Rdna2Format::SMEM: {
                // A load reads SBASE (src[0]: a pair, or the buffer forms' V# quad) and SOFFSET
                // (src[1]) and redefines N consecutive SDATA SGPRs; the effects table gives both
                // widths. Anything not a load -> bail. (This read only SBASE's first two words, so
                // the last two words of a buffer load's V# were not seen as reads.)
                const uint32_t n = rdna2_smem_load_dwords(in.opcode);
                if (!n) return block(in, "unmodelled-smem");
                if (R >= in.src[0].value &&
                    R < in.src[0].value + static_cast<int>(rdna2_smem_address_dwords(in.opcode)))
                    return block(in, "smem-sbase");   // SBASE read
                // SOFFSET is one dword and is consumed as an address, never as a lane mask.
                if ((in.src[1].kind == OperandKind::SGPR ||
                     in.src[1].kind == OperandKind::Special) &&
                    in.src[1].value == R && !data_read_ok)
                    return block(in, "smem-soffset");   // SOFFSET read
                if (R >= in.dst.value && R < in.dst.value + (int)n)
                    continue;   // redefined: this path is dead
                break;
            }
            default: break;
        }
        return std::nullopt;
    } while (false);
    return true;
}

std::optional<bool> CfgScalarMergeRoutes::descriptors() {
    do {
        switch (in.fmt) {
            case Rdna2Format::MIMG:
                // Image packets fully decode both scalar descriptor operands: the eight-dword T#/U#
                // at src[1] and the four-dword S# at src[2]. Some opcodes do not consume the sampler,
                // but accounting for both ranges is deliberately conservative and lets the proof scan
                // through every MIMG without needing a second opcode inventory.
                if (in.src[1].kind != OperandKind::SGPR || in.src[2].kind != OperandKind::SGPR)
                    return false;
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
                     in.src[2].kind == OperandKind::Special) &&
                    in.src[2].value == R)
                    return false;
                break;
            default: break;
        }
        return std::nullopt;
    } while (false);
    return true;
}

std::optional<bool> CfgScalarMergeRoutes::flat_vintrp() {
    do {
        switch (in.fmt) {
            case Rdna2Format::FLAT:
                // FLAT/GLOBAL/SCRATCH packets keep every data/address operand in VGPRs except
                // SADDR, the optional scalar address pair at src[1]. NULL/off (125) reads no
                // scalar register. An LDS transfer also observes M0 implicitly, so keep that
                // uncommon form fail-closed. Account for the complete SADDR pair conservatively;
                // this liveness proof does not imply that the memory operation is translatable.
                if (in.flat_lds) return false;
                if ((in.src[1].kind == OperandKind::SGPR ||
                     in.src[1].kind == OperandKind::Special) &&
                    in.src[1].value != 125 && (in.src[1].value == R || in.src[1].value + 1 == R))
                    return false;
                break;
            case Rdna2Format::VINTRP:
                // CONFIDENCE: HIGH -- AMD 70648 10.4.2/Table 53: P1/P2/MOV read VGPRs
                // and implicit M0, never an ordinary scalar word. EXEC still predicates them;
                // M0 is excluded above. Reserved opcodes and inconsistent decoded forms prove
                // nothing. This only extends the ordinary scalar/VCC-word liveness question.
                if (R < 0 || R > 107 || in.opcode > 2 || in.len_dwords != 1 || in.n_src != 1 ||
                    in.dst.kind != OperandKind::VGPR || in.dst.value < 0 || in.dst.value > 255 ||
                    in.src[0].kind != OperandKind::VGPR || in.src[0].value < 0 ||
                    in.src[0].value > 255 || in.vintrp_attr > 32 || in.vintrp_chan > 3 ||
                    (in.opcode == 2 && in.src[0].value > 2) || in.has_modifier || in.has_sdwa ||
                    in.has_dpp || in.has_literal || in.sdst.kind != OperandKind::None ||
                    in.words[0] !=
                        (0xc8000000u | (in.opcode << 16u) |
                         (static_cast<uint32_t>(in.dst.value) << 18u) | (in.vintrp_attr << 10u) |
                         (in.vintrp_chan << 8u) | static_cast<uint32_t>(in.src[0].value)))
                    return block(in, "unmodelled-vintrp");
                break;
            default: break;
        }
        return std::nullopt;
    } while (false);
    return true;
}

}   // namespace prosper::gpu::detail

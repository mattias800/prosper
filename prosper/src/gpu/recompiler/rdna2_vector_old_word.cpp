#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <cstdint>

namespace prosper::gpu {
namespace {
bool buffer_atomic(uint32_t opcode) {
    return opcode == 0x30 || opcode == 0x32 || opcode == 0x33 ||
           (opcode >= 0x35 && opcode <= 0x3b) || opcode == 0x3f || opcode == 0x40 ||
           opcode == 0x50 || opcode == 0x5a;
}
bool image_atomic(uint32_t opcode) {
    return opcode == 0x0f || opcode == 0x11 || opcode == 0x12 || (opcode >= 0x14 && opcode <= 0x1a);
}
} // namespace

uint32_t vector_provenance_source_span(const Rdna2Inst& in, uint32_t source) {
    if (source >= in.n_src || in.src[source].kind != OperandKind::VGPR) return 0;
    if (in.fmt != Rdna2Format::DS) return rdna2_vgpr_source_span(in, source);
    // CONFIDENCE: HIGH -- the decoder reserves four words for every DS field as storage. The
    // admitted lowerings consume only these actual inputs; reserved fields grant no dependency.
    switch (in.opcode) {
        case 0x35:   // SWIZZLE
        case 0x36:
        case 0x37:
        case 0x38:
        case 0x76:
        case 0x77:
        case 0xfe:
        case 0xff: return source == 0 ? 1 : 0;
        case kDsOpcodeBpermuteB32:
        case 0x00:
        case 0x05:
        case 0x06:
        case 0x07:
        case 0x08:
        case 0x09:
        case 0x0a:
        case 0x0b:
        case 0x0d:
        case 0x12:
        case 0x13:
        case 0x20:
        case 0x2d: return source < 2 ? 1 : 0;
        case 0x3d:
        case 0x3e:
        case 0xb1: return 0;   // implicit M0/TID inputs
        case 0xb0: return source == 1 ? 1 : 0;
        case 0x4d:
        case 0xde:
        case 0xdf:
            return source == 0   ? 1
                   : source == 1 ? (in.opcode == 0x4d   ? 2
                                    : in.opcode == 0xde ? 3
                                                        : 4)
                                 : 0;
        case 0x0e:
        case 0x0f:
        case 0x4e:
            if (source == 0) return 1;
            if (source == 2 && (in.literal & 0xffu) == ((in.literal >> 8) & 0xffu)) return 0;
            return in.opcode == 0x4e ? 2 : 1;
        default: return rdna2_vgpr_source_span(in, source);
    }
}

bool vector_write_leaves_old_word(const Rdna2Inst& in, bool native_interpolation) {
    // CONFIDENCE: HIGH -- native P1/NOP publish no ordinary value. GLC-clear atomics retain
    // VDATA; their conservative decoder write spans describe storage, not a new generation.
    return (in.fmt == Rdna2Format::VOP1 && in.opcode == 0) ||
           (native_interpolation && in.fmt == Rdna2Format::VINTRP && in.opcode == 0) ||
           (in.fmt == Rdna2Format::MUBUF && buffer_atomic(in.opcode) && !in.mubuf_glc) ||
           (in.fmt == Rdna2Format::MIMG && image_atomic(in.opcode) && !in.mimg_glc);
}

bool vector_exec_is_full_after_write(const Rdna2Inst& in, bool full_before, uint32_t wave_size) {
    if (in.fmt != Rdna2Format::SOP1 || in.n_src != 1 || in.src[0].kind != OperandKind::InlineInt)
        return false;
    if (in.opcode == kSop1OpcodeMovB64 && in.dst.value == 126 && in.src[0].value == -1) return true;
    // A low-word reset covers the whole mask only in a proved Wave32 producing stage.
    if (wave_size == 32 && in.opcode == kSop1OpcodeMovB32 && in.dst.value == 126 &&
        in.src[0].value == -1)
        return true;
    // CONFIDENCE: HIGH -- OR_SAVEEXEC with zero preserves the exact old EXEC. Its saved scalar
    // destination does not itself grant mask authority. Other logical writes remain unproved.
    return full_before && in.opcode == 0x25 && in.src[0].value == 0;
}

bool vector_write_depends_on_old_word(const Rdna2Inst& in, bool full_exec) {
    // CONFIDENCE: HIGH -- these are actual old-word reads in prosper's admitted lowerings,
    // including implicit accumulators. Full EXEC does not replace an unwritten half or lane.
    if (!full_exec || (in.sdwa_dst_sel != 6 && in.sdwa_dst_unused == 2)) return true;
    if (in.has_dpp && (in.dpp_row_mask != 15 || in.dpp_bank_mask != 15 ||
                       (!in.dpp_bound_ctrl && in.dpp_ctrl >= 0x111 && in.dpp_ctrl <= 0x11f)))
        return true;
    switch (in.fmt) {
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
        case Rdna2Format::VOP3:
        case Rdna2Format::VOP3P: {
            // The opcode's own answer is the effects table's: an accumulator, a selected-lane or
            // M0-relative destination, or a 16-bit result (which leaves the other half). An
            // encoding the table does not describe -- 0x13c, which prosper lowers through its
            // packed FMAC accumulator service although it is no architectural e64 form -- keeps
            // the old word conservatively.
            const Rdna2Answer keeps = rdna2_valu_result_keeps_old_vdst(in.fmt, in.opcode);
            if (keeps == Rdna2Answer::Unknown) return true;
            const Rdna2InstructionEffects& effects = rdna2_instruction_effects(in);
            // Only SDWA DWORD + UNUSED_PAD (PRESERVE returned above) zero-fills a 16-bit result's
            // other half; the plain and sub-dword forms keep it.
            if (effects.vdst_half && !effects.accumulates)
                return !(in.has_sdwa && in.sdwa_dst_sel == 6);
            if (keeps == Rdna2Answer::Yes) return true;
            // CONFIDENCE: HIGH -- the remaining sub-dword SDWA destinations are those whose
            // prosper lowering inserts the selected field into the old word even under
            // UNUSED_PAD: v_mov_b32 and v_cvt_u32/i32_f32 (VOP1) and v_cndmask_b32 (VOP2). This
            // is a property of prosper's SDWA lowering, not of the instruction, so it stays here.
            if (in.sdwa_dst_sel != 6)
                return in.fmt == Rdna2Format::VOP1
                           ? in.opcode == 1 || in.opcode == 7 || in.opcode == 8
                           : in.fmt == Rdna2Format::VOP2 && in.opcode == 1;
            return false;
        }
        case Rdna2Format::MIMG:
            // The compute image-atomic service preserves VDST for out-of-bounds coordinates.
            // Keeping its old-word dependency is conservative for the native fragment service.
            return image_atomic(in.opcode);
        default: return false;
    }
}
}   // namespace prosper::gpu

#include "gpu/execute/original_graphics_stage_effects.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <algorithm>
#include <array>

namespace prosper::gpu {
namespace {
// Effects knowledge is separate from shader admission. These known register-only operations
// cannot write a guest allocation; unsupported operations are NOT assumed harmless merely
// because rdna2_instruction_may_write_memory's default happens to return false.
bool known_register_effects(const Rdna2Inst& in) {
    switch (in.fmt) {
        case Rdna2Format::SOP1:
            return in.opcode == kSop1OpcodeMovB32 || in.opcode == kSop1OpcodeMovB64 ||
                   in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64 ||
                   in.opcode == kSop1OpcodeNotB32 || in.opcode == 0x08 || in.opcode == 0x0a ||
                   in.opcode == kSop1OpcodeBrevB32 || in.opcode == kSop1OpcodeBcnt1I32B64 ||
                   in.opcode == kSop1OpcodeFf1I32B64 || in.opcode == kSop1OpcodeFlbitI32B32 ||
                   in.opcode == kSop1OpcodeFlbitI32B64 || in.opcode == kSop1OpcodeGetpcB64 ||
                   in.opcode == kSop1OpcodeAndSaveexecB64;
        case Rdna2Format::SOP2:
            return in.opcode == kSop2OpcodeAddU32 || in.opcode == kSop2OpcodeAddI32 ||
                   in.opcode == kSop2OpcodeAddcU32 || in.opcode == kSop2OpcodeCselectB32 ||
                   in.opcode == kSop2OpcodeCselectB64 || in.opcode == kSop2OpcodeAndB32 ||
                   in.opcode == kSop2OpcodeAndB64 || in.opcode == kSop2OpcodeOrB32 ||
                   in.opcode == kSop2OpcodeOrB64;
        case Rdna2Format::SOPK:
            return in.opcode == kSopkOpcodeMovkI32 || in.opcode == kSopkOpcodeCmovkI32;
        case Rdna2Format::SOPC: return in.opcode <= 0x0f || in.opcode == 0x12 || in.opcode == 0x13;
        case Rdna2Format::VOP1:
            // CVT_F32_U32 (gfx10 opcode6) changes only the destination register/FP state.
            return in.opcode == 1 || in.opcode == 2 || in.opcode == 6 || in.opcode == 0x2a ||
                   in.opcode == 0x2e || in.opcode == 0x33;
        case Rdna2Format::VOP2:
            // ADD/MUL/SUB_F32, LSHRREV_B32 and AND_B32 are explicit register-only effects.
            // Opcode identity is checked against the canonical ORIGINAL instruction below.
            return in.opcode == 3 || in.opcode == 4 || in.opcode == 8 || in.opcode == 0x16 ||
                   in.opcode == 0x1b;
        case Rdna2Format::VOP3:
            return in.opcode == 0x360 || in.opcode == 0x365 || in.opcode == 0x366;
        case Rdna2Format::VOPC:
            return (in.opcode >= 0xc0 && in.opcode <= 0xc7) ||
                   (in.opcode >= 0xd0 && in.opcode <= 0xd7);
        default: return false;
    }
}

const char* effect_gap(const Rdna2Inst& in, ShaderProgramStage stage,
                       OriginalGraphicsStageEffects& result) {
    switch (in.fmt) {
        case Rdna2Format::SMEM:
            // RDNA2 scalar DWORD/x2/x4/x8/x16 direct and V# loads. This does not claim that the
            // scalar-bank execution recipe already implements every width or descriptor origin.
            if (in.opcode <= 4 || (in.opcode >= 8 && in.opcode <= 12)) {
                result.guest_memory_reads = true;
                return nullptr;
            }
            return "original-stage-scalar-memory-effect-unimplemented";
        case Rdna2Format::MUBUF:
            if (in.mubuf_lds) return "original-stage-buffer-lds-effect-unimplemented";
            if (rdna2_instruction_may_write_memory(in))
                return "original-stage-memory-effect-not-read-only";
            result.guest_memory_reads = true;
            return nullptr;
        case Rdna2Format::MTBUF:
        case Rdna2Format::MIMG:
            if (rdna2_instruction_may_write_memory(in))
                return "original-stage-memory-effect-not-read-only";
            result.guest_memory_reads = true;
            return nullptr;
        case Rdna2Format::FLAT:
            if (in.flat_lds) return "original-stage-flat-lds-effect-unimplemented";
            if (rdna2_instruction_may_write_memory(in))
                return "original-stage-memory-effect-not-read-only";
            result.guest_memory_reads = true;
            return nullptr;
        case Rdna2Format::DS:
            // Even LDS-only operations need their own complete effect contract. In particular
            // the old may-write helper does not classify GDS stores/atomics at all.
            return "original-stage-ds-gds-effect-unimplemented";
        case Rdna2Format::VINTRP:
            return stage == ShaderProgramStage::Fragment && in.opcode <= 2
                       ? nullptr
                       : "original-stage-interpolation-effect-unimplemented";
        case Rdna2Format::EXP:
            if (stage == ShaderProgramStage::Fragment && in.exp_target <= 9) {
                result.attachment_exports = true;
                return nullptr;
            }
            if (stage == ShaderProgramStage::Vertex &&
                ((in.exp_target >= 12 && in.exp_target <= 15) || in.exp_target >= 32))
                return nullptr;
            return "original-stage-export-target-unimplemented";
        case Rdna2Format::SOPP:
            if (in.opcode == 0 || (in.opcode == 1 && in.is_end) || in.opcode == 0x0c)
                return nullptr;
            // ISA Table72 and an actual gfx1030 LLVM roundtrip identify opcode32 as instruction
            // prefetch, NOT clause. It reads instruction cache state, not a guest data writer.
            if (in.opcode == 0x20 && in.simm16 >= 1 && in.simm16 <= 3) return nullptr;
            return "original-stage-control-or-program-effect-unimplemented";
        case Rdna2Format::Unknown: return "original-stage-unknown-encoding";
        default:
            return known_register_effects(in) ? nullptr
                                              : "original-stage-register-effect-unimplemented";
    }
}
}   // namespace

OriginalGraphicsStageEffects original_graphics_stage_effects(const std::vector<uint32_t>& original,
                                                             const std::vector<Rdna2Inst>& full,
                                                             ShaderProgramStage stage) {
    OriginalGraphicsStageEffects result;
    result.source_words = &original;
    result.stage = stage;
    const auto refuse = [&](const char* why, uint32_t pc) {
        result.architectural_end = false;
        result.rejection = std::string(why) + ":pc=" + std::to_string(pc);
        return result;
    };
    if (stage != ShaderProgramStage::Vertex && stage != ShaderProgramStage::Fragment)
        return refuse("original-stage-kind-unimplemented", UINT32_MAX);
    if (original.empty() || original.size() > UINT32_MAX || full.empty())
        return refuse("original-stage-inventory-unavailable", UINT32_MAX);
    size_t pc = 0;
    for (const auto& supplied : full) {
        if (result.architectural_end || supplied.synthetic_terminator || supplied.pc != pc ||
            pc >= original.size())
            return refuse("original-stage-original-inventory-incomplete", uint32_t(pc));
        // Decode a bounded padded LOCAL window so the decoder's buffer-end length clamp cannot
        // turn a missing literal/NSA/second DWORD into a complete instruction. Padding is only
        // length discovery, never guest backing, and is rejected before any effect is certified.
        std::array<uint32_t, 6> window{};
        std::copy_n(original.data() + pc, std::min(window.size(), original.size() - pc),
                    window.data());
        const auto canonical = rdna2_decode_one(window.data(), window.size());
        if (!canonical.len_dwords || canonical.len_dwords > original.size() - pc)
            return refuse("original-stage-truncated-instruction", uint32_t(pc));
        if (supplied.fmt != canonical.fmt || supplied.opcode != canonical.opcode ||
            supplied.len_dwords != canonical.len_dwords || supplied.is_end != canonical.is_end ||
            supplied.has_literal != canonical.has_literal ||
            supplied.literal != canonical.literal ||
            !std::equal(std::begin(supplied.words), std::end(supplied.words),
                        std::begin(canonical.words)))
            return refuse("original-stage-source-inventory-mismatch", uint32_t(pc));
        if (const char* gap = effect_gap(canonical, stage, result))
            return refuse(gap, uint32_t(pc));
        pc += canonical.len_dwords;
        result.consumed_dwords = uint32_t(pc);
        ++result.instruction_count;
        result.architectural_end =
            canonical.fmt == Rdna2Format::SOPP && canonical.opcode == 1 && canonical.is_end;
    }
    if (!result.architectural_end || pc != original.size())
        return refuse("original-stage-original-inventory-incomplete", uint32_t(pc));
    return result;
}
}   // namespace prosper::gpu

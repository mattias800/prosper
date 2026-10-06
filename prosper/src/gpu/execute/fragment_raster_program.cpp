#include "gpu/execute/fragment_raster_program.hpp"
#include "gpu/recompiler/fragment_packet_quad_swizzle.hpp"
#include "gpu/recompiler/rdna2_waitcnt.hpp"
#include <algorithm>
#include <array>
#include <bitset>

namespace prosper::gpu {
FragmentRasterProgram fragment_raster_program(const std::vector<Rdna2Inst>& instructions,
                                              uint32_t original_words,
                                              PixelSystemInputMapping inputs,
                                              uint32_t user_presence) {
    FragmentRasterProgram result;
    const auto refuse = [&](const char* reason, uint32_t pc) {
        FragmentRasterProgram failed;
        failed.rejection = std::string(reason) + ":pc=" + std::to_string(pc);
        return failed;
    };
    if ((inputs.ena | inputs.addr) & ~0xffffu)
        return refuse("fragment-raster-original-system-input-bits-unimplemented", UINT32_MAX);
    // RDNA2's ADDR packing reserves disabled fields too. Only the actual enabled position
    // fields are supplied by this recipe; unused barycentrics/system fields remain absent.
    constexpr uint32_t widths[16]{2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::array<uint8_t, 256> defined{};   // 0=absent, 1=live only, 2=every genuine backed quad lane
    uint32_t reg = 0;
    for (uint32_t field = 0; field < 16; ++field) {
        if (!(inputs.addr & (1u << field))) continue;
        if ((inputs.ena & (1u << field)) && field >= 8 && field <= 9) {
            result.positions.push_back({reg, 5u + field - 8u});
            defined[reg] = 2;
        }
        reg += widths[field];
    }
    enum class Mask : uint8_t { Absent, IncomingLive, WholeQuad };
    Mask exec = Mask::IncomingLive;
    std::array<Mask, 105>
        aliases{};   // ordinary SGPR pair starts, never VCC/EXEC or system defaults
    std::bitset<106> words;
    for (uint32_t user = 0; user < 32; ++user)
        if (user_presence & (1u << user)) words.set(user);
    const auto scalar = [&](const Operand& operand, const Rdna2Inst& in) {
        if (operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat)
            return true;
        if (operand.kind == OperandKind::Literal) return in.has_literal;
        if (operand.kind != OperandKind::SGPR || operand.value < 0 || operand.value >= 106 ||
            !words.test(operand.value))
            return false;
        // A saved mask is not ordinary numeric input in a quad-local packing proof.
        const auto r = static_cast<uint32_t>(operand.value);
        return (r >= aliases.size() || aliases[r] == Mask::Absent) &&
               (!r || aliases[r - 1] == Mask::Absent);
    };
    const auto mask = [&](const Operand& operand) {
        // Scalar sources decode EXEC as Special; ordinary saved pairs remain SGPR aliases.
        if (operand.kind == OperandKind::Special && operand.value == 126) return exec;
        if (operand.kind != OperandKind::SGPR) return Mask::Absent;
        return operand.value >= 0 && operand.value < 105 ? aliases[operand.value] : Mask::Absent;
    };
    const auto overwrite_scalar = [&](uint32_t first, uint32_t count) {
        for (uint32_t r = first; r < first + count; ++r) {
            words.reset(r);
            if (r < aliases.size()) aliases[r] = Mask::Absent;
            if (r) aliases[r - 1] = Mask::Absent;
        }
    };
    bool saved_live = false, expanded = false, consumed_helper = false, ended = false;
    uint32_t exports = 0, next_pc = 0;
    for (const auto& in : instructions) {
        const auto fail = [&](const char* reason) { return refuse(reason, in.pc); };
        if (ended || in.pc != next_pc || in.synthetic_terminator || !in.len_dwords ||
            in.pc >= original_words || in.len_dwords > original_words - in.pc || in.has_modifier ||
            in.has_sdwa || in.has_dpp || in.clamp || in.omod ||
            std::any_of(std::begin(in.src_abs), std::end(in.src_abs),
                        [](bool value) { return value; }) ||
            std::any_of(std::begin(in.src_neg), std::end(in.src_neg),
                        [](bool value) { return value; }))
            return fail("fragment-raster-original-form-unimplemented");
        next_pc += in.len_dwords;
        if (in.fmt == Rdna2Format::SOP1 && (in.opcode == kSop1OpcodeMovB64 || in.opcode == 0x0a)) {
            auto source = mask(in.src[0]);
            if (source == Mask::Absent) return fail("fragment-raster-mask-source-unproved");
            if (in.opcode == 0x0a) {
                source = Mask::WholeQuad;
                expanded = true;
            }
            if (in.dst.kind != OperandKind::SGPR)
                return fail("fragment-raster-mask-destination-unimplemented");
            if (in.dst.value == 126)
                exec = source;
            else if (in.dst.value >= 0 && in.dst.value < 105) {
                overwrite_scalar(in.dst.value, 2);
                aliases[in.dst.value] = source;
                saved_live |= source == Mask::IncomingLive;
            } else
                return fail("fragment-raster-mask-destination-unimplemented");
        } else if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 &&
                   in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value < 106) {
            if (!scalar(in.src[0], in)) return fail("fragment-raster-scalar-source-unproved");
            overwrite_scalar(in.dst.value, 1);
            words.set(in.dst.value);
        } else if (in.fmt == Rdna2Format::VOP1 && in.opcode == 1 &&
                   in.dst.kind == OperandKind::VGPR && in.dst.value >= 0 && in.dst.value < 256) {
            const auto& source = in.src[0];
            const bool vector = source.kind == OperandKind::VGPR && source.value >= 0 &&
                                source.value < 256 &&
                                defined[source.value] >= (exec == Mask::WholeQuad ? 2u : 1u);
            if (!vector && !scalar(source, in))
                return fail("fragment-raster-vector-source-unproved");
            defined[in.dst.value] =
                std::max(defined[in.dst.value], uint8_t(exec == Mask::WholeQuad ? 2 : 1));
        } else if (in.fmt == Rdna2Format::DS && !packet_quad_swizzle_gap(in)) {
            if (exec != Mask::WholeQuad || defined[in.src[0].value] != 2)
                return fail("fragment-raster-genuine-quad-helper-source-unproved");
            defined[in.dst.value] = 2;
            consumed_helper = true;
        } else if (in.fmt == Rdna2Format::EXP) {
            if (++exports != 1 || exec != Mask::IncomingLive || in.exp_target || in.exp_en != 15 ||
                in.exp_compr || !(in.words[0] & (1u << 11)) || !(in.words[0] & (1u << 12)))
                return fail("fragment-raster-original-live-export-recipe-unimplemented");
            for (const auto& source : in.src)
                if (source.kind != OperandKind::VGPR || source.value < 0 || source.value >= 256 ||
                    !defined[source.value])
                    return fail("fragment-raster-export-source-unproved");
        } else if (in.fmt == Rdna2Format::SOPP && in.opcode == 1 && in.is_end) {
            ended = in.pc + in.len_dwords == original_words;
            if (!ended) return fail("fragment-raster-original-end-unproved");
        } else if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x20 &&
                   !rdna2_is_valid_instruction_prefetch(in)) {
            return fail("fragment-raster-prefetch-mode-unimplemented");
        } else if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x21) {
            return fail("fragment-raster-clause-unimplemented");
        } else if (!(in.fmt == Rdna2Format::SOPP &&
                     (in.opcode == 0 || rdna2_is_valid_instruction_prefetch(in) ||
                      (in.opcode == 0x0c && !rdna2_waitcnt_execution_gap(uint16_t(in.simm16))))))
            return fail("fragment-raster-quad-local-program-unimplemented");
    }
    uint32_t failure_pc = UINT32_MAX;
    if (const auto* gap = packet_quad_swizzle_completion_gap(instructions, failure_pc))
        return refuse(gap, failure_pc);
    if (!ended || exports != 1 || !saved_live || !expanded || !consumed_helper)
        return refuse("fragment-raster-saved-live-helper-recipe-unproved", UINT32_MAX);
    return result;
}
}   // namespace prosper::gpu
